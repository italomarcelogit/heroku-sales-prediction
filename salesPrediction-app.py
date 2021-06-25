import numpy
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

st.set_page_config(layout="wide")
st.sidebar.header('Sales Forecast - Setup:')
df = pd.read_csv('vendas.csv')
listaClientes = df.CLIENTE.sort_values(ascending=True).unique().tolist()
listaVendedores = df.VENDEDOR.sort_values(ascending=True).unique().tolist()
listaProdutos = df.PRODUTO.sort_values(ascending=True).unique().tolist()


def setup_creator():
    # clientes
    clientes = ['Select']
    clientes.extend(listaClientes)
    clientes = tuple(clientes)
    cliente = st.sidebar.selectbox('Consumers:', clientes)
    # vendedores
    vendedores = ['Select']
    if cliente != 'Select':
        vendedores.extend(df[df.CLIENTE == cliente].VENDEDOR.sort_values(ascending=True).unique().tolist())
    else:
        vendedores.extend(df.VENDEDOR.sort_values(ascending=True).unique().tolist())
    vendedores = tuple(vendedores)
    vendedor = st.sidebar.selectbox('Sellers:', vendedores)


    # produtos
    produtos = ['Select']
    if cliente != 'Select':
        produtos.extend(df[df.CLIENTE == cliente].PRODUTO.sort_values(ascending=True).unique().tolist())
    elif vendedor != 'Select':
        produtos.extend(df[df.VENDEDOR == vendedor].PRODUTO.sort_values(ascending=True).unique().tolist())
    else:
        produtos.extend(df.PRODUTO.sort_values(ascending=True).unique().tolist())
    produtos = tuple(produtos)
    produto = st.sidebar.selectbox('Product', produtos)

    mes = st.sidebar.slider(f'Mês:', 1, 12, 6)

    dados = {
        'EMPRESA': 1,
        'MES': mes
    }
    if cliente != 'Select':
        dados.update({'CLIENTE': cliente})
    if vendedor != 'Select':
        dados.update({'VENDEDOR': vendedor})
    if produto != 'Select':
        dados.update({'PRODUTO': produto})

    return pd.DataFrame(data=dados, index=[0])


def getValues(dataframe, field):
    return dataframe[field].values[0]


def topo():
    st.title("Machine Learning - Sales Forecast")


def previsao(dfp):
    st.subheader('Setup - Params')
    st.write(dfp)

    base = df.copy()

    base['id_cliente'] = base.CLIENTE.apply(lambda x: listaClientes.index(x))
    base['id_produto'] = base.PRODUTO.apply(lambda x: listaProdutos.index(x))
    base['id_vendedor'] = base.VENDEDOR.apply(lambda x: listaVendedores.index(x))

    features = ['EMPRESA']
    campo = []
    valor = []

    try:
        if getValues(dfp, 'MES') != 'Select':
            campo.append('MES')
            valor.append(getValues(dfp, 'MES'))
    except:
        pass

    try:
        if getValues(dfp, 'CLIENTE') != 'Select':
            campo.append('id_cliente')
            valor.append(listaClientes.index(getValues(dfp, 'CLIENTE')))
    except:
        pass

    try:
        if getValues(dfp, 'VENDEDOR') != 'Select':
            campo.append('id_vendedor')
            valor.append(listaVendedores.index(getValues(dfp, 'VENDEDOR')))
    except:
        pass

    try:
        if getValues(dfp, 'PRODUTO') != 'Select':
            campo.append('id_produto')
            valor.append(listaProdutos.index(getValues(dfp, 'PRODUTO')))
    except:
        pass

    target = 'TOTAL'
    features.extend(campo)
    valores = [1]
    valores.extend(valor)
    titulo = 'Sales Forecast - Filtered (Setup)'
    if len(features) == 2:
        titulo = 'Sales Forecast - no filter (data company)'
        e = numpy.ones(12)
        m = numpy.arange(1, 13)

        base2 = base[['EMPRESA', 'MES', 'TOTAL']].groupby(['EMPRESA', 'MES']).sum()
        base = pd.DataFrame(columns=['TOTAL', 'MES', 'EMPRESA'])
        base['EMPRESA'] = e
        base['MES'] = m
        base['TOTAL'] = base2['TOTAL'].values

    X = base[features]
    y = base[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = RandomForestRegressor(max_depth=2, random_state=0)

    clf.fit(X_train, y_train)


    st.subheader(titulo)
    st.write(f"""# US$ {clf.predict([valores])[0]:.2f}""")

dfr = setup_creator()
topo()
previsao(dfr)
