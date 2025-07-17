import streamlit as st
import pandas as pd
import joblib
import pickle
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar as coisas
@st.cache_resource
def load_artifacts():
    clf = joblib.load('modelo_rf.pkl')
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('threshold.txt', 'r') as f:
        threshold = float(f.read())
    with open('importances.pkl', 'rb') as f:
        importances = pickle.load(f)
    return clf, features, threshold, importances
clf, features, threshold, importances = load_artifacts()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embedding_model = load_embedding_model()

# Carregar bases (pré-carregadas no app)
with open('amostra_vagas.json', 'r', encoding='utf-8') as f:
    vagas_data = json.load(f)
with open('amostra_applicants.json', 'r', encoding='utf-8') as f:
    applicants_data = json.load(f)
with open('amostra_prospects.json', 'r', encoding='utf-8') as f:
    prospects_data = json.load(f)

def calcular_similaridade_bin(cv, desc_vaga):
    emb_cv = embedding_model.encode([cv])[0]
    emb_vaga = embedding_model.encode([desc_vaga])[0]
    sim = cosine_similarity([emb_cv], [emb_vaga])[0][0]
    if sim < 0.25:
        bin0, bin1, bin2, bin3 = 1,0,0,0
    elif sim < 0.5:
        bin0, bin1, bin2, bin3 = 0,1,0,0
    elif sim < 0.75:
        bin0, bin1, bin2, bin3 = 0,0,1,0
    else:
        bin0, bin1, bin2, bin3 = 0,0,0,1
    return sim, [bin0, bin1, bin2, bin3]

def preparar_features(cand, vaga, sim_bins):
    row = {
        'cand_nivel_academico': cand.get('formacao_e_idiomas', {}).get('nivel_academico', 'Desconhecido'),
        'cand_area_atuacao': cand.get('informacoes_profissionais', {}).get('area_atuacao', 'Desconhecido'),
        'cand_nivel_ingles': cand.get('formacao_e_idiomas', {}).get('nivel_ingles', 'Desconhecido'),
        'cand_nivel_espanhol': cand.get('formacao_e_idiomas', {}).get('nivel_espanhol', 'Desconhecido'),
        'vaga_nivel_profissional_vaga': vaga.get('perfil_vaga', {}).get('nivel profissional', 'Desconhecido'),
        'vaga_nivel_academico_vaga': vaga.get('perfil_vaga', {}).get('nivel_academico', 'Desconhecido'),
        'vaga_nivel_ingles_vaga': vaga.get('perfil_vaga', {}).get('nivel_ingles', 'Desconhecido'),
        'vaga_areas_atuacao_vaga': vaga.get('perfil_vaga', {}).get('areas_atuacao', 'Desconhecido'),
        'similaridade_cv_vaga_bin_0': sim_bins[0],
        'similaridade_cv_vaga_bin_1': sim_bins[1],
        'similaridade_cv_vaga_bin_2': sim_bins[2],
        'similaridade_cv_vaga_bin_3': sim_bins[3]
    }
    return row

def predict_batch(df_feat):
    X = pd.get_dummies(df_feat)
    print(X.columns)
    print('features do treino:', features)
    X = X.reindex(columns=features, fill_value=0)
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba

st.title("Recomendador de Contratação")

# === 1. Seleção ou criação de vaga ===
st.header("Vaga")
tipo_vaga = st.radio("Escolha:", ['Selecionar vaga existente', 'Criar nova vaga'])

if tipo_vaga == 'Selecionar vaga existente':
    vaga_labels = [
        f"{vid} - {vagas_data[vid]['informacoes_basicas'].get('titulo_vaga', '')}"
        for vid in vagas_data
    ]
    vaga_labels.insert(0, 'Selecione uma vaga')
    selected_label = st.selectbox("Selecione a vaga:", vaga_labels, key='vaga_sel')

    if selected_label != 'Selecione uma vaga':
        vaga_id = selected_label.split(' - ')[0]
        vaga = vagas_data.get(vaga_id, {})
        detalhes = []
        infos_basicas = vaga.get('informacoes_basicas', {})
        perfil = vaga.get('perfil_vaga', {})
        beneficios = vaga.get('beneficios', {})

        for k, v in infos_basicas.items():
            detalhes.append({'Categoria': 'Informações Básicas', 'Campo': k, 'Valor': v})
        for k, v in perfil.items():
            detalhes.append({'Categoria': 'Perfil da Vaga', 'Campo': k, 'Valor': v})
        for k, v in beneficios.items():
            detalhes.append({'Categoria': 'Benefícios', 'Campo': k, 'Valor': v})

        df_vaga = pd.DataFrame(detalhes)
        st.dataframe(df_vaga, hide_index=True)
    else:
        st.info("Selecione uma vaga para visualizar os detalhes.")
else:
    st.subheader("Cadastrar nova vaga")
    titulo = st.text_input("Título da vaga")
    nivel_prof = st.selectbox("Nível Profissional", ["Sênior", "Pleno", "Júnior", "Especialista", "Analista", "Desconhecido"])
    nivel_acad = st.text_input("Nível Acadêmico")
    nivel_ingles = st.text_input("Nível de Inglês")
    area_atuacao = st.text_input("Área de Atuação")
    atividades = st.text_area("Principais Atividades")
    vaga = {
        "perfil_vaga": {
            "nivel profissional": nivel_prof,
            "nivel_academico": nivel_acad,
            "nivel_ingles": nivel_ingles,
            "areas_atuacao": area_atuacao,
            "principais_atividades": atividades
        }
    }

# === 2. Seleção ou upload de candidatos ===
st.header("Candidatos")
modo_candidato = st.radio("Escolha:", ['Selecionar candidatos existentes', 'Upload .csv de novos candidatos'])

candidatos_selecionados = []
if modo_candidato == 'Selecionar candidatos existentes':
    candidatos_ids = list(applicants_data.keys())
    candidatos_escolhidos = st.multiselect("Selecione os candidatos:", candidatos_ids)
    for cid in candidatos_escolhidos:
        candidatos_selecionados.append(applicants_data[cid])
else:
    st.info("Faça upload do CSV no modelo: cand_nivel_academico, cand_area_atuacao, cand_nivel_ingles, cand_nivel_espanhol, cv_pt, cargo_atual, etc.")
    csv_file = st.file_uploader("Upload do arquivo CSV de candidatos", type='csv')
    if csv_file:
        df_new = pd.read_csv(csv_file)
        # Transforma cada linha em dict (no formato esperado)
        for i, row in df_new.iterrows():
            cand = {
                "formacao_e_idiomas": {
                    "nivel_academico": row.get('cand_nivel_academico', 'Desconhecido'),
                    "nivel_ingles": row.get('cand_nivel_ingles', 'Desconhecido'),
                    "nivel_espanhol": row.get('cand_nivel_espanhol', 'Desconhecido')
                },
                "informacoes_profissionais": {
                    "area_atuacao": row.get('cand_area_atuacao', 'Desconhecido')
                },
                "cv_pt": row.get('cv_pt', ''),
                "cargo_atual": row.get('cargo_atual', '')
            }
            candidatos_selecionados.append(cand)

if candidatos_selecionados:
    st.subheader("Candidatos Selecionados")
    df_cands = pd.DataFrame([{
        "ID": idx if modo_candidato == 'Selecionar candidatos existentes' else f"Novo_{i+1}",
        "Nível Acadêmico": c.get('formacao_e_idiomas', {}).get('nivel_academico', ''),
        "Área de Atuação": c.get('informacoes_profissionais', {}).get('area_atuacao', ''),
        "Cargo Atual": c.get('cargo_atual', {}).get('cargo_atual', ''),
    } for i, c in enumerate(candidatos_selecionados) for idx in ([c.get('infos_basicas', {}).get('nome', '')] if modo_candidato == 'Selecionar candidatos existentes' else [f"Novo_{i+1}"])])

    st.dataframe(df_cands, hide_index=True)

    # --- Rodar modelo para todos os candidatos ---
    if st.button("Rodar modelo para todos os candidatos"):
        results = []
        for i, cand in enumerate(candidatos_selecionados):
            cv_pt = cand.get('cv_pt', '')
            desc_vaga = vaga.get('perfil_vaga', {}).get('principais_atividades', '')
            sim, bins = calcular_similaridade_bin(cv_pt, desc_vaga)
            features_dict = preparar_features(cand, vaga, bins)
            features_dict['display_id'] = df_cands.iloc[i]['ID']  # Para manter referência
            results.append(features_dict)
        df_results = pd.DataFrame(results)
        pred, proba = predict_batch(df_results)

        # Adiciona resultados ao DataFrame para exibir
        df_results['Probabilidade (%)'] = (proba * 100).round(1)
        df_results['Aprovado'] = ['Sim' if p == 1 else 'Não' for p in pred]
        df_results['ID'] = df_results['display_id']

        # Ranking: ordenar pela probabilidade decrescente
        df_ranking = df_results.sort_values('Probabilidade (%)', ascending=False)
        st.session_state['df_ranking'] = df_ranking.reset_index(drop=True)
        st.session_state['candidatos_selecionados'] = candidatos_selecionados
        st.success("Modelo rodado! Veja o ranking abaixo.")

        st.markdown("### Ranking dos candidatos para esta vaga")
        st.dataframe(df_ranking[['ID', 'Probabilidade (%)', 'Aprovado']], hide_index=True)

        # Selecionar para ver detalhes
        ids = list(df_ranking['ID'])
        ids.insert(0, 'Selecione um candidato')
        cand_id_sel = st.selectbox("Selecione o candidato para ver detalhes:", ids, key='cand_sel')

        cand = next(
            (c for c in candidatos_selecionados if
            (c.get('infos_basicas', {}).get('nome', '') if modo_candidato == 'Selecionar candidatos existentes'
            else f"Novo_{candidatos_selecionados.index(c)+1}") == cand_id_sel),
            None
        )
        if cand is None:
            st.warning("Candidato não encontrado!")
            st.stop()
        st.json(cand)

        # Mostra também explicabilidade simples
        st.markdown("#### Principais features utilizadas no modelo:")
        importances_sorted = sorted(zip(features, importances), key=lambda x: -x[1])
        st.write(pd.DataFrame(importances_sorted, columns=['Feature', 'Importância']).head(8))
    
    if 'df_ranking' in st.session_state:
        df_ranking = st.session_state['df_ranking']
        candidatos_selecionados = st.session_state['candidatos_selecionados']

        st.markdown("### Ranking dos candidatos para esta vaga")
        st.dataframe(df_ranking[['ID', 'Probabilidade (%)', 'Aprovado']], hide_index=True)

        cand_id_sel = st.selectbox(
            "Selecione o ID do candidato para ver detalhes:",
            df_ranking['ID'],
            key='cand_sel'
        )
        cand = next(
            (c for c in candidatos_selecionados if
            (c.get('infos_basicas', {}).get('nome', '') if modo_candidato == 'Selecionar candidatos existentes'
            else f"Novo_{candidatos_selecionados.index(c)+1}") == cand_id_sel),
            None
        )
        if cand is None:
            st.warning("Candidato não encontrado!")
            st.stop()
        # Monta tabela de detalhes principais do candidato
        ver_json = st.toggle("Ver JSON bruto do candidato")  # Usa st.toggle (Streamlit >=1.25)
        if ver_json:
            st.json(cand)
        else:
            infos_basicas = cand.get('infos_basicas', {})
            formacao = cand.get('formacao_e_idiomas', {})
            profissionais = cand.get('informacoes_profissionais', {})
            cargo = cand.get('cargo_atual', {})
            cv_pt = cand.get('cv_pt', '')

            detalhes = {
                "Nome": infos_basicas.get('nome', ''),
                "Email": infos_basicas.get('email', ''),
                "Área de Atuação": profissionais.get('area_atuacao', ''),
                'Objetivo Profissional': infos_basicas.get('objetivo_profissional',''),
                "Nível Profissional": profissionais.get('nivel_profissional', ''),
                "Tempo de Experiência": profissionais.get('tempo_experiencia', ''),
                "Nível Acadêmico": formacao.get('nivel_academico', ''),
                "Nível Inglês": formacao.get('nivel_ingles', ''),
                "Nível Espanhol": formacao.get('nivel_espanhol', ''),
                "Cargo Atual": cargo.get('cargo_atual', '')
            }

            # Mostra os detalhes em tabela
            st.markdown("#### Detalhes do Candidato")
            st.table(pd.DataFrame(list(detalhes.items()), columns=['Campo', 'Valor']).set_index('Campo'))

            # Mostra CV em campo expansível
            if cv_pt and len(cv_pt.strip()) > 0:
                with st.expander("Ver Currículo (CV)"):
                    st.text_area("Currículo", cv_pt, height=500)
            else:
                st.info("CV não disponível para este candidato.")

            st.markdown("#### Principais features utilizadas no modelo:")
            importances_sorted = sorted(zip(features, importances), key=lambda x: -x[1])
            st.dataframe(pd.DataFrame(importances_sorted, columns=['Feature', 'Importância']).head(10), hide_index=True)
else:
    st.info("Selecione pelo menos um candidato ou faça upload do CSV.")

