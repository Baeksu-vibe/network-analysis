import streamlit as st
import pandas as pd
import re
from itertools import combinations
from collections import Counter
import io
import zipfile
import base64
import tempfile
import os
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date

# 페이지 설정
st.set_page_config(
    page_title="특허 시계열 동시출현빈도 분석 도구",
    page_icon="📊",
    layout="wide"
)

# 앱 제목 및 설명
st.title("특허 시계열 동시출현빈도 분석 도구")
st.markdown("""
이 도구는 특허 데이터에서 다음 항목들의 시계열 동시출현빈도를 분석하여 네트워크 분석용 파일을 생성합니다:
- **IPC 코드** 동시출현빈도 (시계열 분석)
- **발명자** 동시출현빈도 (시계열 분석)
- **출원인** 동시출현빈도 (시계열 분석)

**중심성 지표 포함**:
- **EC (Eigenvector Centrality)**: 고유벡터 중심성
- **BC (Betweenness Centrality)**: 매개 중심성
- **CC (Closeness Centrality)**: 근접 중심성

**시계열 분석 기능**:
- 출원일 기준 시계열 분석
- 최대 3개 구간 비교 분석
- 구간별 네트워크 변화 추적

Gephi와 같은 네트워크 시각화 도구에서 사용할 수 있는 노드와 엣지 파일을 생성합니다.
""")

# 필요한 함수들 정의
def parse_application_date(date_str):
    """
    출원일 문자열을 datetime 객체로 변환합니다.
    다양한 날짜 형식을 지원합니다.
    """
    if pd.isna(date_str) or date_str == '' or date_str is None:
        return None
    
    # 문자열로 변환하고 공백 제거
    date_str = str(date_str).strip()
    
    if date_str == '' or date_str.lower() == 'nan':
        return None
    
    # 다양한 날짜 형식 패턴
    date_formats = [
        '%Y-%m-%d',
        '%Y/%m/%d', 
        '%Y.%m.%d',
        '%Y-%m',
        '%Y/%m',
        '%Y.%m',
        '%Y',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%d.%m.%Y',
        '%m-%d-%Y',
        '%d-%m-%Y'
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            # 년도가 너무 미래나 과거인 경우 필터링
            if 1990 <= parsed_date.year <= 2030:
                return parsed_date
        except ValueError:
            continue
    
    # 숫자만 있는 경우 (예: 20230101)
    if date_str.isdigit():
        if len(date_str) == 8:  # YYYYMMDD
            try:
                parsed_date = datetime.strptime(date_str, '%Y%m%d')
                if 1990 <= parsed_date.year <= 2030:
                    return parsed_date
            except ValueError:
                pass
        elif len(date_str) == 6:  # YYYYMM
            try:
                parsed_date = datetime.strptime(date_str, '%Y%m')
                if 1990 <= parsed_date.year <= 2030:
                    return parsed_date
            except ValueError:
                pass
        elif len(date_str) == 4:  # YYYY
            try:
                parsed_date = datetime.strptime(date_str, '%Y')
                if 1990 <= parsed_date.year <= 2030:
                    return parsed_date
            except ValueError:
                pass
    
    # Excel 날짜 숫자 형식 처리 시도
    try:
        # Excel에서 날짜가 숫자로 저장된 경우
        excel_date = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(excel_date) and 1990 <= excel_date.year <= 2030:
            return excel_date
    except:
        pass
    
    return None

def filter_data_by_period(df, date_column, start_year, end_year):
    """
    지정된 기간으로 데이터를 필터링합니다.
    """
    if date_column not in df.columns:
        st.error(f"'{date_column}' 컬럼이 데이터프레임에 없습니다.")
        return df
    
    # 날짜 파싱
    df_filtered = df.copy()
    df_filtered['parsed_date'] = df_filtered[date_column].apply(parse_application_date)
    
    # 유효한 날짜가 있는 행만 필터링 (None이 아닌 값들만)
    df_filtered = df_filtered[df_filtered['parsed_date'].notna()]
    
    if len(df_filtered) == 0:
        st.warning(f"유효한 날짜 데이터가 없습니다.")
        return pd.DataFrame(columns=df.columns)
    
    # datetime 타입으로 변환
    df_filtered['parsed_date'] = pd.to_datetime(df_filtered['parsed_date'])
    
    # 연도 추출
    df_filtered['year'] = df_filtered['parsed_date'].dt.year
    
    # 기간 필터링
    df_filtered = df_filtered[(df_filtered['year'] >= start_year) & (df_filtered['year'] <= end_year)]
    
    # 임시 컬럼 제거하고 원본 데이터 반환
    return df_filtered.drop(['parsed_date'], axis=1)

def extract_shortened_ipc_codes(ipc_str, code_length=4):
    """
    IPC 코드 문자열에서 개별 IPC 코드들을 추출하고 지정된 길이로 축약합니다.
    code_length: 4(섹션+클래스) 또는 8(서브클래스+메인그룹)
    """
    if pd.isna(ipc_str) or ipc_str == '':
        return []
    
    # 문자열로 변환 (숫자 등 다른 타입이 들어올 경우 대비)
    ipc_str = str(ipc_str)
    
    # 원본 IPC 코드 추출
    codes = []
    if '[' in ipc_str and ']' in ipc_str:
        # 대괄호 안과 밖의 코드를 모두 추출
        outside_brackets = ipc_str.split('[')[0].strip()
        inside_brackets = ipc_str.split('[')[1].split(']')[0].strip()
        
        if outside_brackets:
            codes.append(outside_brackets)
        
        if inside_brackets:
            inside_codes = [code.strip() for code in inside_brackets.split(',')]
            codes.extend(inside_codes)
    else:
        # 대괄호가 없는 경우 단일 IPC 코드로 처리
        codes = [ipc_str.strip()]
    
    # 코드 길이에 따라 변환
    shortened_codes = []
    for code in codes:
        if code_length == 4:
            # B66B-001/34 -> B66B
            match = re.match(r'([A-Z]\d{2}[A-Z])', code)
            if match:
                shortened_codes.append(match.group(1))
        elif code_length == 8:
            # B66B-001/34 -> B66B-001
            match = re.match(r'([A-Z]\d{2}[A-Z]-\d{3})', code)
            if match:
                shortened_codes.append(match.group(1))
    
    return shortened_codes

def extract_entities_from_delimited_string(entity_str, delimiter="|"):
    """
    구분자로 분리된 문자열에서 개별 엔티티들을 추출합니다.
    (발명자, 출원인 등에 사용)
    """
    if pd.isna(entity_str) or entity_str == '':
        return []
    
    # 문자열로 변환
    entity_str = str(entity_str)
    
    # 구분자로 분리하고 공백 제거
    entities = [entity.strip() for entity in entity_str.split(delimiter)]
    
    # 빈 문자열 제거
    entities = [entity for entity in entities if entity]
    
    return entities

def calculate_centrality_measures(edges_df, nodes_df):
    """
    네트워크 중심성 지표들을 계산합니다.
    """
    # NetworkX 그래프 생성
    G = nx.Graph()
    
    # 노드 추가
    for _, node in nodes_df.iterrows():
        G.add_node(node['id'], name=node['Name'], label=node['Label'])
    
    # 엣지 추가 (가중치 포함)
    for _, edge in edges_df.iterrows():
        G.add_edge(edge['Source'], edge['Target'], weight=edge['Weight'])
    
    # 연결된 컴포넌트만 분석 (고립된 노드 제외)
    if G.number_of_edges() == 0:
        # 엣지가 없는 경우 모든 중심성을 0으로 설정
        centrality_dict = {
            'EC': {node: 0.0 for node in G.nodes()},
            'BC': {node: 0.0 for node in G.nodes()},
            'CC': {node: 0.0 for node in G.nodes()}
        }
    else:
        # 가장 큰 연결 컴포넌트 선택
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc)
        
        # 중심성 지표 계산
        try:
            # Eigenvector Centrality (고유벡터 중심성)
            ec = nx.eigenvector_centrality(G_connected, weight='weight', max_iter=1000)
        except:
            # 수렴하지 않는 경우 균등하게 분배
            ec = {node: 1.0/len(G_connected) for node in G_connected.nodes()}
        
        try:
            # Betweenness Centrality (매개 중심성)
            bc = nx.betweenness_centrality(G_connected, weight='weight')
        except:
            bc = {node: 0.0 for node in G_connected.nodes()}
        
        try:
            # Closeness Centrality (근접 중심성)
            cc = nx.closeness_centrality(G_connected, distance='weight')
        except:
            cc = {node: 0.0 for node in G_connected.nodes()}
        
        # 연결되지 않은 노드들에 대해 0 값 할당
        centrality_dict = {
            'EC': {node: ec.get(node, 0.0) for node in G.nodes()},
            'BC': {node: bc.get(node, 0.0) for node in G.nodes()},
            'CC': {node: cc.get(node, 0.0) for node in G.nodes()}
        }
    
    # 노드 데이터프레임에 중심성 지표 추가
    nodes_with_centrality = nodes_df.copy()
    nodes_with_centrality['EC'] = nodes_with_centrality['id'].map(centrality_dict['EC'])
    nodes_with_centrality['BC'] = nodes_with_centrality['id'].map(centrality_dict['BC'])
    nodes_with_centrality['CC'] = nodes_with_centrality['id'].map(centrality_dict['CC'])
    
    # Degree 계산 (연결 수)
    degree_dict = dict(G.degree())
    nodes_with_centrality['Degree'] = nodes_with_centrality['id'].map(degree_dict)
    
    # Weighted Degree 계산 (가중치 합)
    weighted_degree_dict = dict(G.degree(weight='weight'))
    nodes_with_centrality['Weighted_Degree'] = nodes_with_centrality['id'].map(weighted_degree_dict)
    
    return nodes_with_centrality

def calculate_timeseries_stats(df, date_column, entity_column, entity_type="Entity"):
    """
    시계열 통계를 계산합니다.
    """
    if date_column not in df.columns or entity_column not in df.columns:
        return pd.DataFrame()
    
    # 날짜 파싱
    df_temp = df.copy()
    df_temp['parsed_date'] = df_temp[date_column].apply(parse_application_date)
    
    # 유효한 날짜가 있는 행만 필터링 (None이 아닌 값들만)
    df_temp = df_temp[df_temp['parsed_date'].notna()]
    
    if len(df_temp) == 0:
        st.warning(f"유효한 날짜 데이터가 없어 시계열 분석을 수행할 수 없습니다.")
        return pd.DataFrame()
    
    # datetime 타입으로 변환
    df_temp['parsed_date'] = pd.to_datetime(df_temp['parsed_date'])
    df_temp['year'] = df_temp['parsed_date'].dt.year
    
    # 연도별 통계
    yearly_stats = []
    
    for year in sorted(df_temp['year'].unique()):
        year_data = df_temp[df_temp['year'] == year]
        
        # 엔티티 추출
        all_entities = []
        for _, row in year_data.iterrows():
            if entity_type == "IPC":
                entities = extract_shortened_ipc_codes(row[entity_column], code_length=4)
            else:
                entities = extract_entities_from_delimited_string(row[entity_column])
            all_entities.extend(entities)
        
        unique_entities = len(set(all_entities))
        total_patents = len(year_data)
        
        yearly_stats.append({
            'Year': year,
            'Total_Patents': total_patents,
            f'Unique_{entity_type}': unique_entities,
            f'Avg_{entity_type}_per_Patent': len(all_entities) / total_patents if total_patents > 0 else 0
        })
    
    return pd.DataFrame(yearly_stats)

def create_timeseries_visualization(stats_df, entity_type, periods_info=None):
    """
    시계열 시각화를 생성합니다.
    """
    if stats_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'연도별 특허 건수',
            f'연도별 고유 {entity_type} 수',
            f'특허당 평균 {entity_type} 수',
            f'누적 특허 건수'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. 연도별 특허 건수
    fig.add_trace(
        go.Scatter(x=stats_df['Year'], y=stats_df['Total_Patents'], 
                  mode='lines+markers', name='특허 건수', line=dict(color='blue')),
        row=1, col=1
    )
    
    # 2. 연도별 고유 엔티티 수
    fig.add_trace(
        go.Scatter(x=stats_df['Year'], y=stats_df[f'Unique_{entity_type}'], 
                  mode='lines+markers', name=f'고유 {entity_type}', line=dict(color='green')),
        row=1, col=2
    )
    
    # 3. 특허당 평균 엔티티 수
    fig.add_trace(
        go.Scatter(x=stats_df['Year'], y=stats_df[f'Avg_{entity_type}_per_Patent'], 
                  mode='lines+markers', name=f'평균 {entity_type}', line=dict(color='red')),
        row=2, col=1
    )
    
    # 4. 누적 특허 건수
    cumulative_patents = stats_df['Total_Patents'].cumsum()
    fig.add_trace(
        go.Scatter(x=stats_df['Year'], y=cumulative_patents, 
                  mode='lines+markers', name='누적 특허', line=dict(color='purple')),
        row=2, col=2
    )
    
    # 구간 표시 (있는 경우)
    if periods_info:
        colors = ['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)']
        for i, period in enumerate(periods_info):
            if i < len(colors):
                for row in range(1, 3):
                    for col in range(1, 3):
                        fig.add_vrect(
                            x0=period['start'], x1=period['end'],
                            fillcolor=colors[i], opacity=0.3,
                            layer="below", line_width=0,
                            row=row, col=col
                        )
    
    fig.update_layout(
        title=f'{entity_type} 시계열 분석',
        showlegend=True,
        height=600
    )
    
    return fig

def calculate_ipc_cooccurrence_timeseries(df, ipc_column, code_length=4, periods=None):
    """시계열을 고려한 축약된 IPC 코드 간의 동시출현빈도를 계산합니다."""
    
    # IPC 컬럼이 존재하는지 확인
    if ipc_column not in df.columns:
        st.error(f"'{ipc_column}' 컬럼이 데이터프레임에 없습니다.")
        st.write("사용 가능한 컬럼:", df.columns.tolist())
        return {}, {}
    
    # 전체 데이터에 대한 분석
    results = {}
    
    # 전체 기간 분석
    st.info("전체 기간 IPC 코드 분석 중...")
    edges_all, nodes_all = calculate_single_period_ipc(df, ipc_column, code_length, "전체 기간")
    results['전체 기간'] = {'edges': edges_all, 'nodes': nodes_all}
    
    # 구간별 분석
    if periods:
        for period_name, period_info in periods.items():
            st.info(f"{period_name} IPC 코드 분석 중...")
            filtered_df = filter_data_by_period(df, '출원일', period_info['start'], period_info['end'])
            
            if len(filtered_df) > 0:
                edges_period, nodes_period = calculate_single_period_ipc(filtered_df, ipc_column, code_length, period_name)
                results[period_name] = {'edges': edges_period, 'nodes': nodes_period}
            else:
                st.warning(f"{period_name}에 해당하는 데이터가 없습니다.")
                results[period_name] = {'edges': pd.DataFrame(), 'nodes': pd.DataFrame()}
    
    return results

def calculate_single_period_ipc(df, ipc_column, code_length, period_name):
    """단일 기간에 대한 IPC 동시출현빈도를 계산합니다."""
    all_combinations = []
    all_codes = []
    
    if len(df) == 0:
        # 빈 데이터프레임인 경우 빈 결과 반환
        empty_node_df = pd.DataFrame(columns=['id', 'Name', 'Label'])
        empty_edge_df = pd.DataFrame(columns=['Source', 'Target', 'type', 'Weight'])
        return empty_edge_df, empty_node_df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df)
    
    # 데이터프레임을 리스트로 변환하여 인덱싱 문제 방지
    df_reset = df.reset_index(drop=True)
    
    for idx in range(total_rows):
        if idx % 10 == 0:
            progress = min(int((idx / total_rows) * 100), 100)  # 100을 초과하지 않도록 제한
            progress_bar.progress(progress)
            status_text.text(f"{period_name} IPC 코드 처리 중... {idx}/{total_rows} 행 완료 ({progress}%)")
            
        row = df_reset.iloc[idx]
        ipc_str = row[ipc_column]
        ipc_codes = extract_shortened_ipc_codes(ipc_str, code_length)
        
        unique_codes = list(set(ipc_codes))
        all_codes.extend(unique_codes)
        
        if len(unique_codes) >= 2:
            pairs = list(combinations(unique_codes, 2))
            sorted_pairs = [tuple(sorted(pair)) for pair in pairs]
            all_combinations.extend(sorted_pairs)
    
    progress_bar.progress(100)
    status_text.text(f"{period_name} IPC 코드 처리 완료! 총 {len(all_combinations)}개의 조합 생성됨")
    
    # 빈도 계산
    counter = Counter(all_combinations)
    code_counter = Counter(all_codes)
    unique_codes = sorted(list(code_counter.keys()))
    
    # 노드 데이터프레임 생성
    node_df = pd.DataFrame({
        'id': range(len(unique_codes)),
        'Name': unique_codes,
        'Label': unique_codes
    })
    
    # 엣지 데이터 생성
    if unique_codes:
        ipc_to_id = dict(zip(unique_codes, node_df['id']))
        edges_data = []
        for (source, target), weight in counter.items():
            source_id = ipc_to_id[source]
            target_id = ipc_to_id[target]
            edges_data.append((source_id, target_id, 'undirected', weight))
        
        edge_df = pd.DataFrame(edges_data, columns=['Source', 'Target', 'type', 'Weight'])
        
        # 중심성 지표 계산
        if len(edge_df) > 0:
            node_df_with_centrality = calculate_centrality_measures(edge_df, node_df)
        else:
            node_df_with_centrality = node_df
    else:
        edge_df = pd.DataFrame(columns=['Source', 'Target', 'type', 'Weight'])
        node_df_with_centrality = node_df
    
    return edge_df, node_df_with_centrality

def calculate_entity_cooccurrence_timeseries(df, entity_column, entity_type="Entity", periods=None):
    """시계열을 고려한 발명자나 출원인 등의 엔티티 간 동시출현빈도를 계산합니다."""
    
    if entity_column not in df.columns:
        st.error(f"'{entity_column}' 컬럼이 데이터프레임에 없습니다.")
        st.write("사용 가능한 컬럼:", df.columns.tolist())
        return {}
    
    results = {}
    
    # 전체 기간 분석
    st.info(f"전체 기간 {entity_type} 분석 중...")
    edges_all, nodes_all = calculate_single_period_entity(df, entity_column, entity_type, "전체 기간")
    results['전체 기간'] = {'edges': edges_all, 'nodes': nodes_all}
    
    # 구간별 분석
    if periods:
        for period_name, period_info in periods.items():
            st.info(f"{period_name} {entity_type} 분석 중...")
            filtered_df = filter_data_by_period(df, '출원일', period_info['start'], period_info['end'])
            
            if len(filtered_df) > 0:
                edges_period, nodes_period = calculate_single_period_entity(filtered_df, entity_column, entity_type, period_name)
                results[period_name] = {'edges': edges_period, 'nodes': nodes_period}
            else:
                st.warning(f"{period_name}에 해당하는 데이터가 없습니다.")
                results[period_name] = {'edges': pd.DataFrame(), 'nodes': pd.DataFrame()}
    
    return results

def calculate_single_period_entity(df, entity_column, entity_type, period_name):
    """단일 기간에 대한 엔티티 동시출현빈도를 계산합니다."""
    all_combinations = []
    all_entities = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df)
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            progress = min(int((idx / total_rows) * 100), 100)  # 100을 초과하지 않도록 제한
            progress_bar.progress(progress)
            status_text.text(f"{period_name} {entity_type} 처리 중... {idx}/{total_rows} 행 완료 ({progress}%)")
            
        entity_str = row[entity_column]
        entities = extract_entities_from_delimited_string(entity_str, delimiter="|")
        
        unique_entities = list(set(entities))
        all_entities.extend(unique_entities)
        
        if len(unique_entities) >= 2:
            pairs = list(combinations(unique_entities, 2))
            sorted_pairs = [tuple(sorted(pair)) for pair in pairs]
            all_combinations.extend(sorted_pairs)
    
    progress_bar.progress(100)
    status_text.text(f"{period_name} {entity_type} 처리 완료! 총 {len(all_combinations)}개의 조합 생성됨")
    
    # 빈도 계산
    counter = Counter(all_combinations)
    entity_counter = Counter(all_entities)
    unique_entities = sorted(list(entity_counter.keys()))
    
    # 노드 데이터프레임 생성
    node_df = pd.DataFrame({
        'id': range(len(unique_entities)),
        'Name': unique_entities,
        'Label': unique_entities
    })
    
    # 엣지 데이터 생성
    if unique_entities:
        entity_to_id = dict(zip(unique_entities, node_df['id']))
        edges_data = []
        for (source, target), weight in counter.items():
            source_id = entity_to_id[source]
            target_id = entity_to_id[target]
            edges_data.append((source_id, target_id, 'undirected', weight))
        
        edge_df = pd.DataFrame(edges_data, columns=['Source', 'Target', 'type', 'Weight'])
        
        # 중심성 지표 계산
        if len(edge_df) > 0:
            node_df_with_centrality = calculate_centrality_measures(edge_df, node_df)
        else:
            node_df_with_centrality = node_df
    else:
        edge_df = pd.DataFrame(columns=['Source', 'Target', 'type', 'Weight'])
        node_df_with_centrality = node_df
    
    return edge_df, node_df_with_centrality

def apply_label_mapping(node_df, mapping_file, code_length=4):
    """매핑 테이블을 사용하여 노드 데이터의 Label 컬럼을 업데이트합니다."""
    try:
        try:
            mapping_df = pd.read_excel(mapping_file)
        except Exception as e:
            st.error(f"매핑 테이블 로드 오류: {e}")
            return node_df
        
        st.write("매핑 테이블 컬럼:", mapping_df.columns.tolist())
        
        ipc_column_candidates = ['IPC', 'Code', 'ipc', 'code', 'Symbol', 'symbol', 'Name']
        label_column_candidates = ['Label', 'Description', 'label', 'description', 'Title', 'title', 'Definition', 'definition']
        
        ipc_column_name = None
        label_column_name = None
        
        for col in mapping_df.columns:
            if any(cand.lower() in col.lower() for cand in ipc_column_candidates):
                sample_values = mapping_df[col].dropna().astype(str).head(3).tolist()
                
                if code_length == 4 and any(re.match(r'^[A-Z]\d{2}[A-Z]', str(val)) for val in sample_values):
                    ipc_column_name = col
                elif code_length == 8 and any(re.match(r'^[A-Z]\d{2}[A-Z]-\d{3}', str(val)) for val in sample_values):
                    ipc_column_name = col
            
            if any(cand.lower() in col.lower() for cand in label_column_candidates):
                label_column_name = col
        
        if not ipc_column_name or not label_column_name:
            st.warning("매핑 테이블에서 컬럼을 자동으로 식별할 수 없습니다.")
            
            ipc_column_name = st.selectbox(
                "IPC 코드가 있는 컬럼을 선택하세요:", 
                options=mapping_df.columns, 
                index=0 if mapping_df.columns.any() else None
            )
            
            label_column_name = st.selectbox(
                "레이블 정보가 있는 컬럼을 선택하세요:", 
                options=[col for col in mapping_df.columns if col != ipc_column_name], 
                index=0 if len(mapping_df.columns) > 1 else None
            )
        
        if ipc_column_name and label_column_name:
            mapping_df[ipc_column_name] = mapping_df[ipc_column_name].astype(str)
            
            if mapping_df[ipc_column_name].duplicated().any():
                st.warning("매핑 테이블에 중복된 IPC 코드가 있습니다. 각 코드의 첫 번째 항목만 사용합니다.")
                mapping_df = mapping_df.drop_duplicates(subset=[ipc_column_name], keep='first')
            
            ipc_to_label = dict(zip(mapping_df[ipc_column_name], mapping_df[label_column_name]))
            
            node_df['OriginalLabel'] = node_df['Label']
            node_df['Label'] = node_df['Name'].map(ipc_to_label)
            node_df.loc[node_df['Label'].isna(), 'Label'] = node_df.loc[node_df['Label'].isna(), 'Name']
            
            mapped_count = (node_df['Label'] != node_df['OriginalLabel']).sum()
            st.success(f"총 {len(node_df)}개 노드 중 {mapped_count}개 노드의 레이블이 업데이트되었습니다.")
            
            node_df = node_df.drop(columns=['OriginalLabel'])
            
            return node_df
        else:
            st.warning("필요한 컬럼을 찾을 수 없어 레이블 매핑을 건너뜁니다.")
            return node_df
    
    except Exception as e:
        st.error(f"레이블 매핑 중 오류 발생: {e}")
        return node_df

def create_zip_file(files_dict):
    """여러 파일을 압축하여 다운로드 가능한 링크를 생성합니다."""
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "patent_network_files.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename, content in files_dict.items():
                content_bytes = content.encode('utf-8')
                zipf.writestr(filename, content_bytes)
        
        with open(zip_path, 'rb') as f:
            bytes_data = f.read()
        
        b64 = base64.b64encode(bytes_data).decode()
        return f'<a href="data:application/zip;base64,{b64}" download="patent_network_files.zip">다운로드: 특허 네트워크 분석 파일</a>'

def get_download_link(df, filename):
    """데이터프레임을 CSV 파일로 변환하여 다운로드 링크를 생성합니다."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">다운로드: {filename}</a>'
    return href

def get_readme_content(results_dict, periods_info=None):
    """README 파일 내용을 생성합니다."""
    readme = """# 특허 시계열 네트워크 분석 파일 설명

## 중심성 지표 설명
- **EC (Eigenvector Centrality)**: 고유벡터 중심성 - 중요한 노드들과 연결된 노드의 중요도
- **BC (Betweenness Centrality)**: 매개 중심성 - 다른 노드들 사이의 최단 경로에 위치하는 정도
- **CC (Closeness Centrality)**: 근접 중심성 - 다른 모든 노드들과의 평균 거리의 역수
- **Degree**: 연결 수 - 직접 연결된 노드의 개수
- **Weighted_Degree**: 가중 연결 수 - 연결 강도를 고려한 연결 수

## 시계열 분석 정보
"""
    
    if periods_info:
        readme += "### 분석 구간\n"
        for period_name, period_info in periods_info.items():
            readme += f"- **{period_name}**: {period_info['start']}년 ~ {period_info['end']}년\n"
        readme += "\n"
    
    readme += """
## 파일 구조
각 분석 항목(IPC 코드, 발명자, 출원인)에 대해 다음과 같은 파일들이 생성됩니다:
- 전체 기간 분석 파일
- 구간별 분석 파일 (설정한 구간이 있는 경우)

## Gephi 사용 방법
1. Gephi를 실행하고 새 프로젝트 생성
2. 데이터 연구실(Data Laboratory) 탭 선택
3. '노드 테이블 가져오기' 클릭 후 *_nodes.csv 파일 불러오기
4. '엣지 테이블 가져오기' 클릭 후 *_edges.csv 파일 불러오기
5. '개요(Overview)' 탭으로 이동하여 네트워크 시각화 및 분석
6. 노드 크기나 색상을 중심성 지표에 따라 조정하여 시각화
7. 시계열 비교를 위해 여러 시기의 파일을 별도로 분석
"""
    return readme

def display_top_pairs(edges_df, nodes_df, pair_type="쌍"):
    """상위 동시출현 쌍을 표시하는 함수"""
    if len(edges_df) == 0:
        st.warning(f"분석할 {pair_type} 데이터가 없습니다.")
        return
    
    st.subheader(f"동시출현빈도 상위 10개 {pair_type}")
    
    top_edges = edges_df.sort_values(by='Weight', ascending=False).head(10)
    
    for idx, row in top_edges.iterrows():
        source_name = nodes_df.loc[nodes_df['id'] == row['Source'], 'Name'].values[0]
        target_name = nodes_df.loc[nodes_df['id'] == row['Target'], 'Name'].values[0]
        source_label = nodes_df.loc[nodes_df['id'] == row['Source'], 'Label'].values[0]
        target_label = nodes_df.loc[nodes_df['id'] == row['Target'], 'Label'].values[0]
        
        st.write(f"**{source_name}** ({source_label}) ↔ **{target_name}** ({target_label}): {row['Weight']}회")

def display_top_centrality_nodes(nodes_df, centrality_column, pair_type="노드", top_n=10):
    """중심성 지표 상위 노드들을 표시하는 함수"""
    if centrality_column not in nodes_df.columns:
        st.warning(f"{centrality_column} 컬럼이 없습니다.")
        return
    
    st.subheader(f"{centrality_column} 상위 {top_n}개 {pair_type}")
    
    top_nodes = nodes_df.sort_values(by=centrality_column, ascending=False).head(top_n)
    
    display_df = top_nodes[['Name', 'Label', centrality_column]].copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df)

def display_period_comparison(results_dict, analysis_type):
    """구간별 비교 분석을 표시하는 함수"""
    if len(results_dict) <= 1:
        return
    
    st.subheader(f"{analysis_type} 구간별 비교")
    
    # 구간별 통계 수집
    comparison_data = []
    for period_name, result in results_dict.items():
        nodes_df = result['nodes']
        edges_df = result['edges']
        
        if len(nodes_df) > 0:
            avg_ec = nodes_df['EC'].mean() if 'EC' in nodes_df.columns else 0
            avg_bc = nodes_df['BC'].mean() if 'BC' in nodes_df.columns else 0
            avg_cc = nodes_df['CC'].mean() if 'CC' in nodes_df.columns else 0
            
            comparison_data.append({
                '구간': period_name,
                '노드 수': len(nodes_df),
                '엣지 수': len(edges_df),
                '평균 EC': round(avg_ec, 4),
                '평균 BC': round(avg_bc, 4),
                '평균 CC': round(avg_cc, 4),
                '최대 가중치': edges_df['Weight'].max() if len(edges_df) > 0 else 0
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)

    # 메인 애플리케이션 로직
def main():
    # 사이드바 설정
    st.sidebar.header("설정")
    
    # 파일 업로드 섹션
    st.sidebar.subheader("1. 데이터 파일 업로드")
    uploaded_file = st.sidebar.file_uploader("Excel 특허 데이터 파일을 업로드하세요", type=["xlsx", "xls"])
    
    # 분석 옵션 선택
    st.sidebar.subheader("2. 분석 옵션 선택")
    analyze_ipc = st.sidebar.checkbox("IPC 코드 분석", value=True)
    analyze_inventor = st.sidebar.checkbox("발명자 분석", value=True)
    analyze_applicant = st.sidebar.checkbox("출원인 분석", value=True)
    
    # 중심성 분석 옵션
    st.sidebar.subheader("3. 중심성 분석 옵션")
    calculate_centrality = st.sidebar.checkbox("중심성 지표 계산 (EC, BC, CC)", value=True)
    
    # 시계열 분석 옵션
    st.sidebar.subheader("4. 시계열 분석 설정")
    enable_timeseries = st.sidebar.checkbox("시계열 분석 활성화", value=True)
    
    periods = {}
    if enable_timeseries:
        num_periods = st.sidebar.selectbox("비교 구간 수", [0, 1, 2, 3], index=0)
        
        if num_periods > 0:
            st.sidebar.write("**구간 설정**")
            for i in range(num_periods):
                period_name = st.sidebar.text_input(f"구간 {i+1} 이름", value=f"구간 {i+1}", key=f"period_name_{i}")
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    start_year = st.number_input(f"시작년도", min_value=1990, max_value=2030, value=2000+i*5, key=f"start_year_{i}")
                with col2:
                    end_year = st.number_input(f"종료년도", min_value=1990, max_value=2030, value=2005+i*5, key=f"end_year_{i}")
                
                if start_year <= end_year:
                    periods[period_name] = {'start': start_year, 'end': end_year}
                else:
                    st.sidebar.error(f"{period_name}: 시작년도가 종료년도보다 클 수 없습니다.")
    
    # 매핑 파일 업로드 섹션
    st.sidebar.subheader("5. 레이블 매핑 파일 (선택사항)")
    uploaded_mapping_4chars = st.sidebar.file_uploader("4자리 IPC 코드 매핑 파일 (선택사항)", type=["xlsx", "xls"])
    uploaded_mapping_8chars = st.sidebar.file_uploader("8자리 IPC 코드 매핑 파일 (선택사항)", type=["xlsx", "xls"])
    
    # 처리 버튼
    process_button = st.sidebar.button("분석 시작", type="primary")
    
    # 메인 화면 레이아웃 - 탭 구성
    tab_names = ["분석 결과", "시계열 대시보드"]
    
    if analyze_ipc:
        tab_names.extend(["4자리 IPC 코드", "8자리 IPC 코드"])
    if analyze_inventor:
        tab_names.append("발명자 분석")
    if analyze_applicant:
        tab_names.append("출원인 분석")
    
    # 탭 생성 및 딕셔너리에 저장
    tabs = st.tabs(tab_names)
    tabs_dict = {}
    for i, tab_name in enumerate(tab_names):
        tabs_dict[tab_name] = tabs[i]
    
    if uploaded_file is not None:
        # 파일 로드 및 처리
        with st.spinner('데이터 파일 로드 중...'):
            df = pd.read_excel(uploaded_file)
            st.success(f"Excel 파일을 성공적으로 로드했습니다. 행 수: {len(df)}")
            
            # 출원일 컬럼 확인
            date_columns = [col for col in df.columns if '출원일' in col or 'date' in col.lower() or '일자' in col]
            
            # 데이터 미리보기
            with tabs_dict["분석 결과"]:
                st.subheader("데이터 미리보기")
                st.dataframe(df.head(5))
                
                # 출원일 컬럼 선택
                if enable_timeseries:
                    st.subheader("출원일 컬럼 선택")
                    if date_columns:
                        default_date_index = 0
                    else:
                        default_date_index = 0
                        st.warning("출원일 관련 컬럼을 찾을 수 없습니다. 수동으로 선택해주세요.")
                    
                    selected_date_column = st.selectbox(
                        "출원일이 포함된 컬럼을 선택하세요:",
                        options=df.columns.tolist(),
                        index=default_date_index,
                        key="date_column"
                    )
                
                # 컬럼 선택
                st.subheader("분석 컬럼 선택")
                
                selected_columns = {}
                
                if analyze_ipc:
                    ipc_column_candidates = [col for col in df.columns if 'ipc' in col.lower() or 'code' in col.lower()]
                    default_index = 0
                    
                    if ipc_column_candidates:
                        try:
                            first_candidate = ipc_column_candidates[0]
                            if first_candidate in df.columns:
                                default_index = int(df.columns.get_loc(first_candidate))
                            else:
                                default_index = 0
                        except:
                            default_index = 0
                    
                    selected_columns['ipc'] = st.selectbox(
                        "IPC 코드가 포함된 컬럼을 선택하세요:",
                        options=df.columns.tolist(),
                        index=default_index,
                        key="ipc_column"
                    )
                
                if analyze_inventor:
                    inventor_column_candidates = [col for col in df.columns if '발명자' in col or 'inventor' in col.lower() or '개발자' in col]
                    default_index = 0
                    
                    if inventor_column_candidates:
                        try:
                            first_candidate = inventor_column_candidates[0]
                            if first_candidate in df.columns:
                                default_index = int(df.columns.get_loc(first_candidate))
                            else:
                                default_index = 0
                        except:
                            default_index = 0
                    
                    selected_columns['inventor'] = st.selectbox(
                        "발명자가 포함된 컬럼을 선택하세요:",
                        options=df.columns.tolist(),
                        index=default_index,
                        key="inventor_column"
                    )
                
                if analyze_applicant:
                    applicant_column_candidates = [col for col in df.columns if '출원인' in col or 'applicant' in col.lower() or '지원자' in col]
                    default_index = 0
                    
                    if applicant_column_candidates:
                        try:
                            first_candidate = applicant_column_candidates[0]
                            if first_candidate in df.columns:
                                default_index = int(df.columns.get_loc(first_candidate))
                            else:
                                default_index = 0
                        except:
                            default_index = 0
                    
                    selected_columns['applicant'] = st.selectbox(
                        "출원인이 포함된 컬럼을 선택하세요:",
                        options=df.columns.tolist(),
                        index=default_index,
                        key="applicant_column"
                    )
        
        if process_button:
            # 출원일 컬럼 설정
            if enable_timeseries:
                df['출원일'] = df[selected_date_column]
            
            # 분석 결과를 저장할 딕셔너리
            results = {}
            
            with tabs_dict["분석 결과"]:
                st.header("분석 진행 중...")
                
                # IPC 코드 분석
                if analyze_ipc:
                    st.subheader("IPC 코드 시계열 분석 중...")
                    
                    # 4자리 IPC 코드
                    if enable_timeseries:
                        results['ipc_4'] = calculate_ipc_cooccurrence_timeseries(df, selected_columns['ipc'], code_length=4, periods=periods)
                    else:
                        edges_4, nodes_4 = calculate_single_period_ipc(df, selected_columns['ipc'], 4, "전체 기간")
                        results['ipc_4'] = {'전체 기간': {'edges': edges_4, 'nodes': nodes_4}}
                    
                    # 8자리 IPC 코드
                    if enable_timeseries:
                        results['ipc_8'] = calculate_ipc_cooccurrence_timeseries(df, selected_columns['ipc'], code_length=8, periods=periods)
                    else:
                        edges_8, nodes_8 = calculate_single_period_ipc(df, selected_columns['ipc'], 8, "전체 기간")
                        results['ipc_8'] = {'전체 기간': {'edges': edges_8, 'nodes': nodes_8}}
                    
                    # 레이블 매핑 적용
                    if uploaded_mapping_4chars is not None:
                        st.subheader("4자리 IPC 코드에 매핑 테이블 적용 중...")
                        for period_name in results['ipc_4'].keys():
                            results['ipc_4'][period_name]['nodes'] = apply_label_mapping(
                                results['ipc_4'][period_name]['nodes'], uploaded_mapping_4chars, code_length=4)
                    
                    if uploaded_mapping_8chars is not None:
                        st.subheader("8자리 IPC 코드에 매핑 테이블 적용 중...")
                        for period_name in results['ipc_8'].keys():
                            results['ipc_8'][period_name]['nodes'] = apply_label_mapping(
                                results['ipc_8'][period_name]['nodes'], uploaded_mapping_8chars, code_length=8)
                
                # 발명자 분석
                if analyze_inventor:
                    st.subheader("발명자 시계열 분석 중...")
                    if enable_timeseries:
                        results['inventor'] = calculate_entity_cooccurrence_timeseries(df, selected_columns['inventor'], "발명자", periods=periods)
                    else:
                        edges_inv, nodes_inv = calculate_single_period_entity(df, selected_columns['inventor'], "발명자", "전체 기간")
                        results['inventor'] = {'전체 기간': {'edges': edges_inv, 'nodes': nodes_inv}}
                
                # 출원인 분석
                if analyze_applicant:
                    st.subheader("출원인 시계열 분석 중...")
                    if enable_timeseries:
                        results['applicant'] = calculate_entity_cooccurrence_timeseries(df, selected_columns['applicant'], "출원인", periods=periods)
                    else:
                        edges_app, nodes_app = calculate_single_period_entity(df, selected_columns['applicant'], "출원인", "전체 기간")
                        results['applicant'] = {'전체 기간': {'edges': edges_app, 'nodes': nodes_app}}
                
                # 분석 결과 요약 표시
                st.header("분석 완료!")
                
                summary_data = []
                if analyze_ipc:
                    for period_name, result in results['ipc_4'].items():
                        summary_data.append({
                            '분석 항목': '4자리 IPC 코드',
                            '구간': period_name,
                            '노드 수': len(result['nodes']),
                            '엣지 수': len(result['edges'])
                        })
                    
                    for period_name, result in results['ipc_8'].items():
                        summary_data.append({
                            '분석 항목': '8자리 IPC 코드',
                            '구간': period_name,
                            '노드 수': len(result['nodes']),
                            '엣지 수': len(result['edges'])
                        })
                
                if analyze_inventor:
                    for period_name, result in results['inventor'].items():
                        summary_data.append({
                            '분석 항목': '발명자',
                            '구간': period_name,
                            '노드 수': len(result['nodes']),
                            '엣지 수': len(result['edges'])
                        })
                
                if analyze_applicant:
                    for period_name, result in results['applicant'].items():
                        summary_data.append({
                            '분석 항목': '출원인',
                            '구간': period_name,
                            '노드 수': len(result['nodes']),
                            '엣지 수': len(result['edges'])
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df)
                
                # 결과 파일 생성
                files_dict = {}
                
                # 각 분석 유형별로 파일 생성
                for analysis_type, analysis_results in results.items():
                    for period_name, result in analysis_results.items():
                        period_suffix = f"_{period_name.replace(' ', '_')}" if period_name != "전체 기간" else ""
                        
                        if analysis_type == 'ipc_4':
                            files_dict[f"ipc_edges_4chars{period_suffix}.csv"] = result['edges'].to_csv(index=False)
                            files_dict[f"ipc_nodes_4chars{period_suffix}.csv"] = result['nodes'].to_csv(index=False)
                        elif analysis_type == 'ipc_8':
                            files_dict[f"ipc_edges_8chars{period_suffix}.csv"] = result['edges'].to_csv(index=False)
                            files_dict[f"ipc_nodes_8chars{period_suffix}.csv"] = result['nodes'].to_csv(index=False)
                        elif analysis_type == 'inventor':
                            files_dict[f"inventor_edges{period_suffix}.csv"] = result['edges'].to_csv(index=False)
                            files_dict[f"inventor_nodes{period_suffix}.csv"] = result['nodes'].to_csv(index=False)
                        elif analysis_type == 'applicant':
                            files_dict[f"applicant_edges{period_suffix}.csv"] = result['edges'].to_csv(index=False)
                            files_dict[f"applicant_nodes{period_suffix}.csv"] = result['nodes'].to_csv(index=False)
                
                files_dict["README.txt"] = get_readme_content(results, periods)
                
                # 다운로드 링크 생성
                download_link = create_zip_file(files_dict)
                st.markdown(download_link, unsafe_allow_html=True)
            
            # 시계열 대시보드 탭
            if enable_timeseries:
                with tabs_dict["시계열 대시보드"]:
                    st.header("시계열 분석 대시보드")
                    
                    # 각 분석 항목별 시계열 통계 생성 및 시각화
                    if analyze_ipc and '출원일' in df.columns:
                        st.subheader("IPC 코드 시계열 분석")
                        ipc_stats = calculate_timeseries_stats(df, '출원일', selected_columns['ipc'], "IPC")
                        if not ipc_stats.empty:
                            periods_info = [{'start': info['start'], 'end': info['end']} for info in periods.values()] if periods else None
                            fig_ipc = create_timeseries_visualization(ipc_stats, "IPC", periods_info)
                            if fig_ipc:
                                st.plotly_chart(fig_ipc, use_container_width=True)
                    
                    if analyze_inventor and '출원일' in df.columns:
                        st.subheader("발명자 시계열 분석")
                        inventor_stats = calculate_timeseries_stats(df, '출원일', selected_columns['inventor'], "발명자")
                        if not inventor_stats.empty:
                            periods_info = [{'start': info['start'], 'end': info['end']} for info in periods.values()] if periods else None
                            fig_inventor = create_timeseries_visualization(inventor_stats, "발명자", periods_info)
                            if fig_inventor:
                                st.plotly_chart(fig_inventor, use_container_width=True)
                    
                    if analyze_applicant and '출원일' in df.columns:
                        st.subheader("출원인 시계열 분석")
                        applicant_stats = calculate_timeseries_stats(df, '출원일', selected_columns['applicant'], "출원인")
                        if not applicant_stats.empty:
                            periods_info = [{'start': info['start'], 'end': info['end']} for info in periods.values()] if periods else None
                            fig_applicant = create_timeseries_visualization(applicant_stats, "출원인", periods_info)
                            if fig_applicant:
                                st.plotly_chart(fig_applicant, use_container_width=True)
            
            # 각 분석 결과 탭에 상세 정보 표시
            if analyze_ipc:
                # 4자리 IPC 코드 분석 결과 탭
                with tabs_dict["4자리 IPC 코드"]:
                    st.header("4자리 IPC 코드 분석 결과")
                    
                    # 구간별 결과 표시
                    for period_name, result in results['ipc_4'].items():
                        st.subheader(f"{period_name} 분석 결과")
                        
                        nodes_df = result['nodes']
                        edges_df = result['edges']
                        
                        if len(nodes_df) > 0:
                            # 중심성 지표 상위 노드들 표시
                            if calculate_centrality and 'EC' in nodes_df.columns:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    display_top_centrality_nodes(nodes_df, 'EC', '4자리 IPC 코드', 5)
                                with col2:
                                    display_top_centrality_nodes(nodes_df, 'BC', '4자리 IPC 코드', 5)
                                with col3:
                                    display_top_centrality_nodes(nodes_df, 'CC', '4자리 IPC 코드', 5)
                            
                            # 상위 동시출현 쌍 표시
                            display_top_pairs(edges_df, nodes_df, "IPC 쌍")
                            
                            # 노드 데이터 표시 (상위 50개)
                            with st.expander(f"{period_name} 노드 데이터 보기"):
                                st.dataframe(nodes_df.head(50))
                            
                            # 엣지 데이터 표시 (상위 20개)
                            with st.expander(f"{period_name} 엣지 데이터 보기"):
                                st.dataframe(edges_df.sort_values(by='Weight', ascending=False).head(20))
                        else:
                            st.warning(f"{period_name}에 분석할 데이터가 없습니다.")
                    
                    # 구간별 비교 분석
                    if len(results['ipc_4']) > 1:
                        display_period_comparison(results['ipc_4'], "4자리 IPC 코드")
                
                # 8자리 IPC 코드 분석 결과 탭
                with tabs_dict["8자리 IPC 코드"]:
                    st.header("8자리 IPC 코드 분석 결과")
                    
                    # 구간별 결과 표시
                    for period_name, result in results['ipc_8'].items():
                        st.subheader(f"{period_name} 분석 결과")
                        
                        nodes_df = result['nodes']
                        edges_df = result['edges']
                        
                        if len(nodes_df) > 0:
                            # 중심성 지표 상위 노드들 표시
                            if calculate_centrality and 'EC' in nodes_df.columns:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    display_top_centrality_nodes(nodes_df, 'EC', '8자리 IPC 코드', 5)
                                with col2:
                                    display_top_centrality_nodes(nodes_df, 'BC', '8자리 IPC 코드', 5)
                                with col3:
                                    display_top_centrality_nodes(nodes_df, 'CC', '8자리 IPC 코드', 5)
                            
                            # 상위 동시출현 쌍 표시
                            display_top_pairs(edges_df, nodes_df, "IPC 쌍")
                            
                            # 노드 데이터 표시 (상위 50개)
                            with st.expander(f"{period_name} 노드 데이터 보기"):
                                st.dataframe(nodes_df.head(50))
                            
                            # 엣지 데이터 표시 (상위 20개)
                            with st.expander(f"{period_name} 엣지 데이터 보기"):
                                st.dataframe(edges_df.sort_values(by='Weight', ascending=False).head(20))
                        else:
                            st.warning(f"{period_name}에 분석할 데이터가 없습니다.")
                    
                    # 구간별 비교 분석
                    if len(results['ipc_8']) > 1:
                        display_period_comparison(results['ipc_8'], "8자리 IPC 코드")
            
            if analyze_inventor:
                # 발명자 분석 결과 탭
                with tabs_dict["발명자 분석"]:
                    st.header("발명자 분석 결과")
                    
                    # 구간별 결과 표시
                    for period_name, result in results['inventor'].items():
                        st.subheader(f"{period_name} 분석 결과")
                        
                        nodes_df = result['nodes']
                        edges_df = result['edges']
                        
                        if len(nodes_df) > 0:
                            # 중심성 지표 상위 노드들 표시
                            if calculate_centrality and 'EC' in nodes_df.columns:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    display_top_centrality_nodes(nodes_df, 'EC', '발명자', 5)
                                with col2:
                                    display_top_centrality_nodes(nodes_df, 'BC', '발명자', 5)
                                with col3:
                                    display_top_centrality_nodes(nodes_df, 'CC', '발명자', 5)
                            
                            # 상위 동시출현 쌍 표시
                            display_top_pairs(edges_df, nodes_df, "발명자 쌍")
                            
                            # 노드 데이터 표시 (상위 50개)
                            with st.expander(f"{period_name} 노드 데이터 보기"):
                                st.dataframe(nodes_df.head(50))
                            
                            # 엣지 데이터 표시 (상위 20개)
                            with st.expander(f"{period_name} 엣지 데이터 보기"):
                                st.dataframe(edges_df.sort_values(by='Weight', ascending=False).head(20))
                        else:
                            st.warning(f"{period_name}에 분석할 데이터가 없습니다.")
                    
                    # 구간별 비교 분석
                    if len(results['inventor']) > 1:
                        display_period_comparison(results['inventor'], "발명자")
            
            if analyze_applicant:
                # 출원인 분석 결과 탭
                with tabs_dict["출원인 분석"]:
                    st.header("출원인 분석 결과")
                    
                    # 구간별 결과 표시
                    for period_name, result in results['applicant'].items():
                        st.subheader(f"{period_name} 분석 결과")
                        
                        nodes_df = result['nodes']
                        edges_df = result['edges']
                        
                        if len(nodes_df) > 0:
                            # 중심성 지표 상위 노드들 표시
                            if calculate_centrality and 'EC' in nodes_df.columns:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    display_top_centrality_nodes(nodes_df, 'EC', '출원인', 5)
                                with col2:
                                    display_top_centrality_nodes(nodes_df, 'BC', '출원인', 5)
                                with col3:
                                    display_top_centrality_nodes(nodes_df, 'CC', '출원인', 5)
                            
                            # 상위 동시출현 쌍 표시
                            display_top_pairs(edges_df, nodes_df, "출원인 쌍")
                            
                            # 노드 데이터 표시 (상위 50개)
                            with st.expander(f"{period_name} 노드 데이터 보기"):
                                st.dataframe(nodes_df.head(50))
                            
                            # 엣지 데이터 표시 (상위 20개)
                            with st.expander(f"{period_name} 엣지 데이터 보기"):
                                st.dataframe(edges_df.sort_values(by='Weight', ascending=False).head(20))
                        else:
                            st.warning(f"{period_name}에 분석할 데이터가 없습니다.")
                    
                    # 구간별 비교 분석
                    if len(results['applicant']) > 1:
                        display_period_comparison(results['applicant'], "출원인")
    else:
        with tabs_dict["분석 결과"]:
            st.info("👈 사이드바에서 Excel 특허 데이터 파일을 업로드하고 분석 옵션을 선택한 후 '분석 시작' 버튼을 클릭하세요.")
            st.info("필요한 경우 IPC 코드 레이블 매핑 파일도 함께 업로드할 수 있습니다.")
            
            st.markdown("""
            ### 📋 데이터 형식 요구사항
            
            **출원일 컬럼**: 다양한 날짜 형식 지원
            - 예시: `2023-01-15`, `2023/01/15`, `2023.01.15`, `20230115`, `2023-01`, `2023`
            
            **발명자 컬럼**: 여러 발명자는 `|` 구분자로 분리
            - 예시: `홍길동|김철수|이영희`
            
            **출원인 컬럼**: 여러 출원인은 `|` 구분자로 분리  
            - 예시: `삼성전자|LG전자`
            
            **IPC 코드**: 기존 형식 유지
            - 예시: `B66B-001/34[H01L-021/00,G06F-015/16]`
            
            ### 📊 중심성 지표 설명
            
            **EC (Eigenvector Centrality)**: 고유벡터 중심성
            - 중요한 노드들과 연결된 노드의 중요도를 측정
            - 값이 높을수록 영향력 있는 노드들과 연결되어 있음
            
            **BC (Betweenness Centrality)**: 매개 중심성
            - 다른 노드들 사이의 최단 경로에 위치하는 정도
            - 값이 높을수록 네트워크에서 중개 역할이 큼
            
            **CC (Closeness Centrality)**: 근접 중심성
            - 다른 모든 노드들과의 평균 거리의 역수
            - 값이 높을수록 다른 모든 노드들에 빠르게 접근 가능
            
            ### ⏰ 시계열 분석 기능
            
            **구간별 비교 분석**: 최대 3개 구간 설정 가능
            - 각 구간별로 네트워크 구조 변화 추적
            - 시기별 중심성 지표 변화 분석
            
            **시계열 대시보드**: 연도별 통계 시각화
            - 특허 출원 트렌드 분석
            - 고유 엔티티 수 변화 추적
            - 구간별 하이라이트 표시
            
            **출력 파일**: 구간별 독립 파일 생성
            - 각 구간에 대한 별도의 노드/엣지 파일
            - Gephi에서 시계열 비교 분석 가능
            """)
        
        # 시계열 대시보드 탭 (파일 업로드 전에도 설명 표시)
        with tabs_dict["시계열 대시보드"]:
            st.header("시계열 분석 대시보드")
            if uploaded_file is None:
                st.info("데이터를 업로드하고 분석을 시작하면 시계열 차트가 여기에 표시됩니다.")
                
                st.markdown("""
                ### 시계열 대시보드 기능
                
                1. **연도별 특허 건수**: 시간에 따른 특허 출원 추세
                2. **고유 엔티티 수**: 각 연도별 고유한 IPC/발명자/출원인 수
                3. **평균 엔티티 수**: 특허당 평균 IPC/발명자/출원인 수
                4. **누적 특허 건수**: 시간에 따른 누적 특허 수
                5. **구간 하이라이트**: 설정한 비교 구간을 차트에 표시
                """)

# 앱 실행
if __name__ == "__main__":
    main()