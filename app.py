
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_obesidade.pkl")

@st.cache_data
def carregar_dados():
    return pd.read_csv("obesity_tratado.csv")

pacote_modelo = carregar_modelo()
df = carregar_dados()

modelo = pacote_modelo["modelo"]
colunas_modelo = pacote_modelo["colunas_modelo"]
mapa_classes = pacote_modelo["mapa_classes"]

# ======================================================
# Preparação da base para visualização em português
# ======================================================

df_viz = df.copy()

mapa_genero = {
    "Female": "Feminino",
    "Male": "Masculino"
}

mapa_sim_nao = {
    "yes": "Sim",
    "no": "Não"
}

mapa_transporte = {
    "Automobile": "Automóvel",
    "Motorbike": "Moto",
    "Bike": "Bicicleta",
    "Public_Transportation": "Transporte público",
    "Walking": "Caminhada"
}

mapa_frequencia = {
    "no": "Não",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre"
}

df_viz["Gênero"] = df_viz["Gender"].map(mapa_genero)
df_viz["Histórico familiar"] = df_viz["family_history"].map(mapa_sim_nao)
df_viz["Consome alimentos calóricos"] = df_viz["FAVC"].map(mapa_sim_nao)
df_viz["Fuma"] = df_viz["SMOKE"].map(mapa_sim_nao)
df_viz["Monitora calorias"] = df_viz["SCC"].map(mapa_sim_nao)
df_viz["Transporte"] = df_viz["MTRANS"].map(mapa_transporte)
df_viz["Consumo entre refeições"] = df_viz["CAEC"].map(mapa_frequencia)
df_viz["Consumo de álcool"] = df_viz["CALC"].map(mapa_frequencia)

ordem_obesidade = [
    "Peso insuficiente",
    "Peso normal",
    "Sobrepeso nível I",
    "Sobrepeso nível II",
    "Obesidade tipo I",
    "Obesidade tipo II",
    "Obesidade tipo III"
]

ordem_grupo = [
    "Abaixo ou peso normal",
    "Sobrepeso",
    "Obesidade"
]

# ======================================================
# Cabeçalho do app
# ======================================================

st.title("🏥 Sistema Preditivo de Nível de Obesidade")

st.write(
    "Aplicação desenvolvida para auxiliar a equipe médica na classificação "
    "do nível de obesidade com base em características físicas, hábitos alimentares "
    "e estilo de vida."
)

aba1, aba2, aba3 = st.tabs([
    "🔎 Predição",
    "📊 Painel Analítico",
    "🤖 Sobre o Modelo"
])

# ======================================================
# Aba 1 - Predição
# ======================================================

with aba1:
    st.header("Predição do nível de obesidade")

    st.write(
        "Preencha as informações abaixo para que o modelo estime o nível de obesidade "
        "com base no perfil informado."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        gender_label = st.selectbox("Gênero", ["Feminino", "Masculino"])
        gender = {"Feminino": "Female", "Masculino": "Male"}[gender_label]

        age = st.number_input("Idade", min_value=10, max_value=100, value=25)
        height = st.number_input("Altura em metros", min_value=1.20, max_value=2.30, value=1.70, step=0.01)
        weight = st.number_input("Peso em kg", min_value=30.0, max_value=250.0, value=75.0, step=0.5)

    with col2:
        family_history_label = st.selectbox("Possui histórico familiar de excesso de peso?", ["Sim", "Não"])
        family_history = {"Sim": "yes", "Não": "no"}[family_history_label]

        favc_label = st.selectbox("Consome alimentos calóricos com frequência?", ["Sim", "Não"])
        favc = {"Sim": "yes", "Não": "no"}[favc_label]

        fcvc = st.selectbox(
            "Consumo de vegetais nas refeições",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Baixo",
                2: "Moderado",
                3: "Alto"
            }[x]
        )

        ncp = st.selectbox(
            "Número de refeições principais por dia",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "1 refeição",
                2: "2 refeições",
                3: "3 refeições",
                4: "4 ou mais refeições"
            }[x]
        )

    with col3:
        caec_label = st.selectbox("Costuma comer entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        caec = {
            "Não": "no",
            "Às vezes": "Sometimes",
            "Frequentemente": "Frequently",
            "Sempre": "Always"
        }[caec_label]

        smoke_label = st.selectbox("Fuma?", ["Sim", "Não"])
        smoke = {"Sim": "yes", "Não": "no"}[smoke_label]

        ch2o = st.selectbox(
            "Consumo diário de água",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Baixo",
                2: "Moderado",
                3: "Alto"
            }[x]
        )

        scc_label = st.selectbox("Monitora calorias diariamente?", ["Sim", "Não"])
        scc = {"Sim": "yes", "Não": "no"}[scc_label]

    col4, col5, col6 = st.columns(3)

    with col4:
        faf = st.selectbox(
            "Frequência de atividade física",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Nenhuma",
                1: "Baixa",
                2: "Moderada",
                3: "Alta"
            }[x]
        )

    with col5:
        tue = st.selectbox(
            "Tempo usando dispositivos tecnológicos",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Baixo",
                1: "Moderado",
                2: "Alto"
            }[x]
        )

    with col6:
        calc_label = st.selectbox("Consumo de álcool", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        calc = {
            "Não": "no",
            "Às vezes": "Sometimes",
            "Frequentemente": "Frequently",
            "Sempre": "Always"
        }[calc_label]

        mtrans_label = st.selectbox(
            "Meio de transporte habitual",
            ["Automóvel", "Moto", "Bicicleta", "Transporte público", "Caminhada"]
        )

        mtrans = {
            "Automóvel": "Automobile",
            "Moto": "Motorbike",
            "Bicicleta": "Bike",
            "Transporte público": "Public_Transportation",
            "Caminhada": "Walking"
        }[mtrans_label]

    imc = weight / (height ** 2)

    dados_usuario = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans,
        "IMC": imc
    }])

    dados_usuario_encoded = pd.get_dummies(dados_usuario)

    for coluna in colunas_modelo:
        if coluna not in dados_usuario_encoded.columns:
            dados_usuario_encoded[coluna] = 0

    dados_usuario_encoded = dados_usuario_encoded[colunas_modelo]

    st.subheader("Resumo do paciente")
    st.write(f"IMC calculado: **{imc:.2f}**")

    if st.button("Realizar predição"):
        previsao = modelo.predict(dados_usuario_encoded)[0]
        previsao_traduzida = mapa_classes.get(previsao, previsao)

        st.success(f"Nível previsto: **{previsao_traduzida}**")
        st.info("Esta predição é um apoio analítico e não substitui avaliação médica profissional.")

# ======================================================
# Aba 2 - Painel Analítico
# ======================================================

with aba2:
    st.header("📊 Painel Analítico sobre Obesidade")

    st.write(
        "Esta visão analítica apresenta os principais padrões encontrados na base, "
        "com foco em apoiar a equipe médica na compreensão dos fatores associados "
        "aos diferentes níveis de obesidade."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de pessoas analisadas", f"{len(df_viz):,}".replace(",", "."))
    col2.metric("Idade média", f"{df_viz['Age'].mean():.1f} anos")
    col3.metric("Peso médio", f"{df_viz['Weight'].mean():.1f} kg")
    col4.metric("IMC médio", f"{df_viz['IMC'].mean():.1f}")

    st.divider()

    st.subheader("Filtros da análise")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        filtro_genero = st.multiselect(
            "Gênero",
            options=sorted(df_viz["Gênero"].dropna().unique()),
            default=sorted(df_viz["Gênero"].dropna().unique())
        )

    with col_f2:
        filtro_grupo = st.multiselect(
            "Grupo de classificação corporal",
            options=ordem_grupo,
            default=ordem_grupo
        )

    with col_f3:
        filtro_historico = st.multiselect(
            "Histórico familiar de excesso de peso",
            options=sorted(df_viz["Histórico familiar"].dropna().unique()),
            default=sorted(df_viz["Histórico familiar"].dropna().unique())
        )

    df_filtrado = df_viz[
        (df_viz["Gênero"].isin(filtro_genero)) &
        (df_viz["Grupo_Obesidade"].isin(filtro_grupo)) &
        (df_viz["Histórico familiar"].isin(filtro_historico))
    ]

    st.caption(f"Registros considerados nos gráficos: {len(df_filtrado)}")

    st.divider()

    st.subheader("1. Distribuição dos níveis de obesidade")

    qtd_obesidade = (
        df_filtrado["Obesity_Descricao"]
        .value_counts()
        .reindex(ordem_obesidade)
        .dropna()
        .reset_index()
    )

    qtd_obesidade.columns = ["Nível de obesidade", "Quantidade de pessoas"]

    fig_obesidade = px.bar(
        qtd_obesidade,
        x="Nível de obesidade",
        y="Quantidade de pessoas",
        text="Quantidade de pessoas",
        title="Quantidade de pessoas por nível de obesidade"
    )

    fig_obesidade.update_traces(textposition="outside")

    fig_obesidade.update_layout(
        xaxis_title="Nível de obesidade",
        yaxis_title="Quantidade de pessoas",
        title_x=0.02
    )

    st.plotly_chart(fig_obesidade, use_container_width=True)

    st.info(
        "Este gráfico mostra como os registros estão distribuídos entre as classes "
        "de peso insuficiente, peso normal, sobrepeso e obesidade."
    )

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("2. Classificação corporal por gênero")

        df_genero = (
            df_filtrado
            .groupby(["Gênero", "Grupo_Obesidade"])
            .size()
            .reset_index(name="Quantidade de pessoas")
        )

        fig_genero = px.bar(
            df_genero,
            x="Gênero",
            y="Quantidade de pessoas",
            color="Grupo_Obesidade",
            barmode="group",
            text="Quantidade de pessoas",
            title="Distribuição dos grupos corporais por gênero",
            labels={
                "Gênero": "Gênero",
                "Quantidade de pessoas": "Quantidade de pessoas",
                "Grupo_Obesidade": "Grupo corporal"
            },
            category_orders={
                "Grupo_Obesidade": ordem_grupo
            }
        )

        fig_genero.update_traces(textposition="outside")

        fig_genero.update_layout(
            xaxis_title="Gênero",
            yaxis_title="Quantidade de pessoas",
            legend_title_text="Grupo corporal",
            title_x=0.02
        )

        st.plotly_chart(fig_genero, use_container_width=True)

    with col_b:
        st.subheader("3. Histórico familiar e obesidade")

        df_historico = (
            df_filtrado
            .groupby(["Histórico familiar", "Grupo_Obesidade"])
            .size()
            .reset_index(name="Quantidade de pessoas")
        )

        fig_historico = px.bar(
            df_historico,
            x="Histórico familiar",
            y="Quantidade de pessoas",
            color="Grupo_Obesidade",
            barmode="group",
            text="Quantidade de pessoas",
            title="Grupo corporal por histórico familiar",
            labels={
                "Histórico familiar": "Histórico familiar de excesso de peso",
                "Quantidade de pessoas": "Quantidade de pessoas",
                "Grupo_Obesidade": "Grupo corporal"
            },
            category_orders={
                "Grupo_Obesidade": ordem_grupo
            }
        )

        fig_historico.update_traces(textposition="outside")

        fig_historico.update_layout(
            xaxis_title="Histórico familiar de excesso de peso",
            yaxis_title="Quantidade de pessoas",
            legend_title_text="Grupo corporal",
            title_x=0.02
        )

        st.plotly_chart(fig_historico, use_container_width=True)

    st.info(
        "Esses gráficos ajudam a observar se existe concentração de sobrepeso ou obesidade "
        "em determinados gêneros ou em pessoas com histórico familiar de excesso de peso."
    )

    st.divider()

    st.subheader("4. IMC por grupo de classificação corporal")

    fig_imc = px.box(
        df_filtrado,
        x="Grupo_Obesidade",
        y="IMC",
        color="Grupo_Obesidade",
        title="Distribuição do IMC por grupo corporal",
        labels={
            "Grupo_Obesidade": "Grupo corporal",
            "IMC": "Índice de Massa Corporal"
        },
        category_orders={
            "Grupo_Obesidade": ordem_grupo
        }
    )

    fig_imc.update_layout(
        xaxis_title="Grupo corporal",
        yaxis_title="IMC",
        showlegend=False,
        title_x=0.02
    )

    st.plotly_chart(fig_imc, use_container_width=True)

    st.info(
        "O boxplot permite visualizar a variação do IMC dentro de cada grupo corporal, "
        "facilitando a comparação entre pessoas com peso normal, sobrepeso e obesidade."
    )

    st.divider()

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("5. Atividade física por grupo corporal")

        df_atividade = (
            df_filtrado
            .groupby(["FAF", "Grupo_Obesidade"])
            .size()
            .reset_index(name="Quantidade de pessoas")
        )

        df_atividade["Frequência de atividade física"] = df_atividade["FAF"].map({
            0: "Nenhuma",
            1: "Baixa",
            2: "Moderada",
            3: "Alta"
        })

        fig_atividade = px.bar(
            df_atividade,
            x="Frequência de atividade física",
            y="Quantidade de pessoas",
            color="Grupo_Obesidade",
            barmode="group",
            text="Quantidade de pessoas",
            title="Frequência de atividade física por grupo corporal",
            labels={
                "Frequência de atividade física": "Frequência de atividade física",
                "Quantidade de pessoas": "Quantidade de pessoas",
                "Grupo_Obesidade": "Grupo corporal"
            },
            category_orders={
                "Frequência de atividade física": ["Nenhuma", "Baixa", "Moderada", "Alta"],
                "Grupo_Obesidade": ordem_grupo
            }
        )

        fig_atividade.update_traces(textposition="outside")

        fig_atividade.update_layout(
            xaxis_title="Frequência de atividade física",
            yaxis_title="Quantidade de pessoas",
            legend_title_text="Grupo corporal",
            title_x=0.02
        )

        st.plotly_chart(fig_atividade, use_container_width=True)

    with col_d:
        st.subheader("6. Meio de transporte habitual")

        df_transporte = (
            df_filtrado
            .groupby(["Transporte", "Grupo_Obesidade"])
            .size()
            .reset_index(name="Quantidade de pessoas")
        )

        fig_transporte = px.bar(
            df_transporte,
            x="Transporte",
            y="Quantidade de pessoas",
            color="Grupo_Obesidade",
            barmode="group",
            text="Quantidade de pessoas",
            title="Grupo corporal por meio de transporte",
            labels={
                "Transporte": "Meio de transporte",
                "Quantidade de pessoas": "Quantidade de pessoas",
                "Grupo_Obesidade": "Grupo corporal"
            },
            category_orders={
                "Grupo_Obesidade": ordem_grupo
            }
        )

        fig_transporte.update_traces(textposition="outside")

        fig_transporte.update_layout(
            xaxis_title="Meio de transporte",
            yaxis_title="Quantidade de pessoas",
            legend_title_text="Grupo corporal",
            title_x=0.02
        )

        st.plotly_chart(fig_transporte, use_container_width=True)

    st.info(
        "Essas análises ajudam a avaliar hábitos de vida relacionados ao sedentarismo, "
        "como prática de atividade física e meio de transporte utilizado."
    )

    st.divider()

    st.subheader("7. Relação entre idade, peso e nível de obesidade")

    fig_dispersao = px.scatter(
        df_filtrado,
        x="Age",
        y="Weight",
        color="Obesity_Descricao",
        hover_data={
            "Age": True,
            "Weight": True,
            "Height": True,
            "IMC": ":.2f",
            "Obesity_Descricao": True
        },
        title="Relação entre idade, peso e nível de obesidade",
        labels={
            "Age": "Idade",
            "Weight": "Peso em kg",
            "Obesity_Descricao": "Nível de obesidade",
            "Height": "Altura",
            "IMC": "IMC"
        },
        category_orders={
            "Obesity_Descricao": ordem_obesidade
        }
    )

    fig_dispersao.update_layout(
        xaxis_title="Idade",
        yaxis_title="Peso em kg",
        legend_title_text="Nível de obesidade",
        title_x=0.02
    )

    st.plotly_chart(fig_dispersao, use_container_width=True)

    st.info(
        "Este gráfico permite observar a relação entre idade, peso e classificação corporal. "
        "Ele ajuda a identificar concentrações de pessoas em faixas de maior risco."
    )

    st.divider()

    st.subheader("Principais leituras para a equipe médica")

    st.markdown("""
    - O painel permite acompanhar a distribuição dos níveis de obesidade da população analisada.
    - O histórico familiar pode ser observado como um fator relevante na concentração de casos de sobrepeso e obesidade.
    - Indicadores como IMC, peso, idade, atividade física e meio de transporte ajudam a contextualizar o risco.
    - A visão analítica apoia a triagem inicial e pode auxiliar a equipe médica na priorização de avaliações clínicas.
    """)

# ======================================================
# Aba 3 - Sobre o Modelo
# ======================================================

with aba3:
    st.header("Sobre o modelo")

    st.write("""
    O modelo final escolhido foi o **Random Forest**, treinado com a base Obesity.csv.

    A base contém informações físicas, comportamentais e de histórico familiar dos indivíduos.
    O objetivo é classificar o nível de obesidade em uma das seguintes categorias:
    """)

    st.write("""
    - Abaixo do peso
    - Peso normal
    - Sobrepeso grau I
    - Sobrepeso grau II
    - Obesidade grau I
    - Obesidade grau II
    - Obesidade grau III
    """)

    st.write("""
    Durante o treinamento, o modelo apresentou acurácia aproximada de **98%** na base de teste,
    superando o requisito mínimo de 75% definido no desafio.
    """)

    st.warning("O sistema é uma ferramenta de apoio à decisão e não substitui avaliação médica.")
