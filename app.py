
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

with aba1:
    st.header("Predição do nível de obesidade")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gênero", ["Female", "Male"])
        age = st.number_input("Idade", min_value=10, max_value=100, value=25)
        height = st.number_input("Altura em metros", min_value=1.20, max_value=2.30, value=1.70, step=0.01)
        weight = st.number_input("Peso em kg", min_value=30.0, max_value=250.0, value=75.0, step=0.5)

    with col2:
        family_history = st.selectbox("Histórico familiar de excesso de peso?", ["yes", "no"])
        favc = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
        fcvc = st.selectbox("Consumo de vegetais nas refeições", [1, 2, 3])
        ncp = st.selectbox("Número de refeições principais por dia", [1, 2, 3, 4])

    with col3:
        caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Fuma?", ["yes", "no"])
        ch2o = st.selectbox("Consumo diário de água", [1, 2, 3])
        scc = st.selectbox("Monitora calorias diariamente?", ["yes", "no"])

    col4, col5, col6 = st.columns(3)

    with col4:
        faf = st.selectbox("Frequência de atividade física", [0, 1, 2, 3])

    with col5:
        tue = st.selectbox("Tempo usando dispositivos tecnológicos", [0, 1, 2])

    with col6:
        calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox(
            "Meio de transporte",
            ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
        )

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

with aba2:
    st.header("Painel analítico sobre obesidade")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de registros", len(df))
    col2.metric("Idade média", f"{df['Age'].mean():.1f}")
    col3.metric("Peso médio", f"{df['Weight'].mean():.1f} kg")
    col4.metric("IMC médio", f"{df['IMC'].mean():.1f}")

    st.subheader("Distribuição dos níveis de obesidade")

    fig_obesidade = px.histogram(
        df,
        x="Obesity_Descricao",
        color="Obesity_Descricao",
        title="Quantidade de pessoas por nível de obesidade"
    )

    fig_obesidade.update_layout(
        xaxis_title="Nível de obesidade",
        yaxis_title="Quantidade",
        showlegend=False
    )

    st.plotly_chart(fig_obesidade, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Obesidade por gênero")

        fig_genero = px.histogram(
            df,
            x="Gender",
            color="Grupo_Obesidade",
            barmode="group",
            title="Grupo de obesidade por gênero"
        )

        st.plotly_chart(fig_genero, use_container_width=True)

    with col_b:
        st.subheader("Histórico familiar")

        fig_historico = px.histogram(
            df,
            x="family_history",
            color="Grupo_Obesidade",
            barmode="group",
            title="Histórico familiar por grupo de obesidade"
        )

        st.plotly_chart(fig_historico, use_container_width=True)

    st.subheader("Relação entre idade, peso e nível de obesidade")

    fig_dispersao = px.scatter(
        df,
        x="Age",
        y="Weight",
        color="Obesity_Descricao",
        hover_data=["Height", "IMC"],
        title="Peso por idade e nível de obesidade"
    )

    st.plotly_chart(fig_dispersao, use_container_width=True)

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
