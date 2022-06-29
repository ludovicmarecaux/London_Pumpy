import streamlit as st
import time
import pandas as pd
from datetime import date
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from joblib import load
from PIL import Image
import pickle
import streamlit.components.v1 as components
pd.set_option("display.max_columns", None)

st.set_page_config(page_title='London Pump.Py',
                   page_icon=':fire:',layout="wide")



latext = r'''
$$ 
\huge
\color{purple}
London Pump.Py
$$ 
'''
st.write(latext)

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    image = Image.open("data_inputs/call_center.jpg")
    st.image(image,use_column_width='auto')

with col3:
    st.write(' ')




df = pd.read_pickle(
    "data_outputs/base_ml.pkl"
)
@st.cache
def To_num(df):
    df.workingday=df.workingday.astype('int64')
    df.school_holidays=df.school_holidays.astype('int64')
    return df
  
def NumPumpsAttending(df, pumps, max):
  liste = list(np.arange(pumps,max+1))
  return df['NumPumpsAttending'].replace(liste, pumps)

# fonction selectionnant les features et la target dans le dataframe
def Select_Column2(df, features):
    return df[features]

# fonction dicotomisant les features de type object
def Object_dummies(df):
    return df.select_dtypes(exclude="object").join(
        pd.get_dummies(df.select_dtypes("object"))
    )

# fonction dicotomisant PumpAvaillable, variable numérique catégorielle
def PumpAvailable_Dummies(df):
    for i in ["PumpAvailable"]:
        dummy3 = pd.get_dummies(df[i], prefix=i)
        df = pd.concat([df, dummy3], axis=1)
        df.drop(["PumpAvailable"], 1, inplace=True)
        return df

def Traitement1(df):
    df = Select_Column(df)
    df = To_num(df)
    return df

def Traitement2(df):
    df = Object_dummies(df)
    df = PumpAvailable_Dummies(df)
    return df


def Select_Column(df):
    return df[
        [
         'DeployedFromLocation',
         'Appliance',
         'PropertyCategory',
         'AddressQualifier',
         'IncidentType',
         'Distance',
         'TotalOfPumpInLondon_Out',
         'Station_Code_of_ressource',
         'IncidentStationGround_Code',
         'PumpAvailable',
         'month',
         'temp',
         'precip',
         'cloudcover',
         'visibility',
         'conditions',
         'workingday',
         'school_holidays',
         'congestion_rate',
         
        ]
    ]



def StandardSC_test(df,df1):
    sc = StandardScaler()
    sc.fit(
        df[
            [
                "Distance",
                "congestion_rate",
                "TotalOfPumpInLondon_Out",
                "temp",
                "precip",
                "cloudcover",
                "visibility",
            ]
        ]
    )
    df1[
        [
            "Distance",
            "congestion_rate",
            "TotalOfPumpInLondon_Out",
            "temp",
            "precip",
            "cloudcover",
            "visibility",
        ]
    ] = sc.transform(
        df1[
            [
                "Distance",
                "congestion_rate",
                "TotalOfPumpInLondon_Out",
                "temp",
                "precip",
                "cloudcover",
                "visibility",
            ]
        ]
    )
    return df1

def reset_button():
    st.session_state["p"] = False
    return



if 'count2' not in st.session_state:
	st.session_state.count2 = 0

# ---- SIDEBAR ----


# pavé pour renseigner le contexte (mois en automatique, météo, congés, trafic)
st.sidebar.header("Context")

today = date.today()
d2 = today.strftime("%B %d, %Y")
st.sidebar.write(d2)

# month = today.month
month_list = range(1,13)
month = st.sidebar.selectbox('Month ?', month_list, index = today.month-1)

temp = st.sidebar.number_input('Temperature ?')

precip = st.sidebar.number_input('Precipitation ?',min_value = 0.00)

cloudcover = st.sidebar.number_input('Cloud cover ?',min_value = 0.0)

visibility = st.sidebar.number_input('Visibility ?')

conditions = st.sidebar.selectbox(
     'Conditions?',
     list(df['conditions'].unique()))
workingday = st.sidebar.selectbox(
     'Today is a working day ? 0: No, 1:Yes',
     list(df['workingday'].unique()))
school_holidays = st.sidebar.selectbox(
     'Today is school holiday day ? 0: No, 1:Yes',
     list(df['school_holidays'].unique()[::-1]),index=0)

congestion_rate = st.sidebar.number_input('Congestion rate ?',min_value = 0.00)


# pavé pour prédiction nombre de véhicules à envoyer
st.sidebar.markdown("""---""")
st.sidebar.header("Number of pumps required prediction")

list_propertycategory=[['-----'],list(df['PropertyCategory'].unique())]
list_propertycategory=list(np.concatenate(list_propertycategory). flat)                                   
PropertyCategory = st.sidebar.selectbox(
     'Which Category ?',
     list_propertycategory)

list_PropertyType=[['-----'],list(df['PropertyType'][df['PropertyCategory']==PropertyCategory].unique())]
list_PropertyType=list(np.concatenate(list_PropertyType). flat)
PropertyType = st.sidebar.selectbox(
    "Which type of Property ?",list_PropertyType)

list_AddressQualifier=[['-----'],list(df['AddressQualifier'].unique())]
list_AddressQualifier=list(np.concatenate(list_AddressQualifier). flat) 
AddressQualifier = st.sidebar.selectbox(
     'Adress Qualifier?',
     list_AddressQualifier)


list_IncidentType=[['-----'],list(df['IncidentType'].unique())]
list_IncidentType=list(np.concatenate(list_IncidentType). flat)
IncidentType = st.sidebar.selectbox(
     'Type of incident?',list_IncidentType) ## Pb : toutes les modalités, même celles non conservées par Christophe (ex : False alarm - Malicious)



if IncidentType == "Fire" :
  list_IncidentCategory=['Secondary Fire', 'AFA', 'Primary Fire', 'Chimney Fire','Late Call']
elif IncidentType == "Prior Arrangement" :
  list_IncidentCategory=['Advice Only', 'Assist other agencies', 'Stand By']  
else:
  list_IncidentCategory=[['-----'],list(df['IncidentCategory'][df['IncidentType']==IncidentType].unique())]
  list_IncidentCategory=list(np.concatenate(list_IncidentCategory). flat)
    
IncidentCategory = st.sidebar.selectbox("Category of Incident",list_IncidentCategory)

how_many_pumps = st.sidebar.button("Predict how many pumps to mobilise")

manual_number_pumps = st.sidebar.checkbox("Need to manually set the number of pumps ?",key='p')


# pavé pour les véhicules mobilisés
st.sidebar.markdown("""---""")
st.sidebar.header("Attendance Time prediction")

IncidentStationGround_Code = st.sidebar.selectbox(
     'Code of the Station ground\'s Incident  ?',
     list(df['IncidentStationGround_Code'].unique()))

PumpAvailable = st.sidebar.selectbox(
     'Number of pump available ?',
     list(df['PumpAvailable'].unique()))

TotalOfPumpInLondon_Out = st.sidebar.selectbox(
     'Number of pump in London out?',
     list(df['TotalOfPumpInLondon_Out'].unique()))

Appliance = st.sidebar.selectbox(
     'Type of the pump?',
     list(df['Appliance'].unique()))

Station_Code_of_ressource = st.sidebar.selectbox(
     'Station code of the resource?',
     list(df['Station_Code_of_ressource'].unique()))

Distance = st.sidebar.number_input(
     'Distance between the incident and the pump ?')

DeployedFromLocation = st.sidebar.selectbox(
     'From where do you send the pump?',
     list(df['DeployedFromLocation'].unique()))



if how_many_pumps:
  if (PropertyCategory == '-----')|(PropertyType == '-----')|(AddressQualifier == '-----')|(IncidentType == '-----')|(IncidentCategory == '-----'):
    st.write(f'''<p style="color:Tomato;">One or more parameters are missing</p>''',unsafe_allow_html=True)
  else:
    df_pumps=df
    df_pumps['NumPumpsAttending'] = NumPumpsAttending( df_pumps, 5, 15)

    df_pumps= df_pumps[ df_pumps['Mobilised_Rank']==1]
    df = df.drop_duplicates(subset=['IncidentNumber'])
    
    df_pumps= df_pumps[( df_pumps['IncidentCategory']!= "False alarm - Good intent") & 
        ( df_pumps['IncidentCategory']!= "False alarm - Malicious") & 
        ( df_pumps['IncidentCategory']!= "No action (not false alarm)")]
        
    colonne_a_conserver = ['PropertyCategory',
                        'PropertyType',
                        'AddressQualifier',
                        'IncidentType',
                        'IncidentCategory',
                        ]
    df_pumps =  df_pumps[colonne_a_conserver]
    
    df_pumps= df_pumps.append({'PropertyType' : PropertyType , 'PropertyCategory' : PropertyCategory, 'AddressQualifier' : AddressQualifier , 'IncidentCategory' : IncidentCategory , 'IncidentType' : IncidentType } , ignore_index=True)
  
    colonne_dummies = ['PropertyCategory',
            'PropertyType',
            'AddressQualifier',
            'IncidentCategory',
            'IncidentType'
            ]
    df_pumps = pd.get_dummies(df_pumps, columns = colonne_dummies)

    df_pumps = df_pumps.astype(float)

    df_pumps_tail = df_pumps.tail(1)
    st.write(df_pumps_tail)

    model=load('data_outputs/ml_nb_pumps/Number_of_Pumps_XGB.joblib')
    
    
    predict_proba = pd.DataFrame(model.predict_proba(df_pumps_tail), columns=('%d pump(s)' % i for i in range(1,6)))
    st.table(predict_proba)

    #st.image(Image.open("/content/drive/MyDrive/Projet Pompier/Github/structure_finale/data_inputs/Pumps.jpg"))

    Predict = model.predict(df_pumps_tail)
    nbre_camion=int(Predict[0])
    NumberOfPumps = predict_proba.idxmax(axis = 1)
    st.write(nbre_camion, 'Pump(s) must be sent')
    st.session_state.count2=nbre_camion


if manual_number_pumps:
  nbre_camion2 = st.slider('Number of pump?', 0, 10,1 )
  st.session_state.count2 = nbre_camion2
  st.write(nbre_camion2, 'Pump(s) must be sent')
 
  



#   st.write("il faut envoyer :",NumberOfPumps[0]) 
 


# Arrivée dans les temps??

df=Traitement1(df)
if 'df1' not in st.session_state:
	st.session_state.df1 = pd.DataFrame({'DeployedFromLocation':pd.Series(dtype='object'), 'Appliance':pd.Series(dtype='object'), 'PropertyCategory':pd.Series(dtype='object'),
       'AddressQualifier':pd.Series(dtype='object'), 'IncidentType':pd.Series(dtype='object'), 'Distance':pd.Series(dtype='float64'),
       'TotalOfPumpInLondon_Out':pd.Series(dtype='int'), 'Station_Code_of_ressource':pd.Series(dtype='object'),
       'IncidentStationGround_Code':pd.Series(dtype='object'), 'PumpAvailable':pd.Series(dtype='int'), 'month':pd.Series(dtype='int'), 'temp':pd.Series(dtype='float64'),
       'precip':pd.Series(dtype='float64'), 'cloudcover':pd.Series(dtype='float64'), 'visibility':pd.Series(dtype='float64'), 'conditions':pd.Series(dtype='object'), 'workingday':pd.Series(dtype='int'),
       'school_holidays':pd.Series(dtype='int'), 'congestion_rate':pd.Series(dtype='float64')})

if 'count' not in st.session_state:
	st.session_state.count = 0


if st.sidebar.button('Add this pump to mobilised list'):
  if st.session_state.count2 == 0:
    st.write (f'''<p style="color:green;">You must indicate the number of pumps first</p>''',unsafe_allow_html=True)
  elif (st.session_state.count2 != 0) & (st.session_state.count < st.session_state.count2):
      new_row=pd.Series(data={'DeployedFromLocation':DeployedFromLocation, 'Appliance':Appliance, 'PropertyCategory':PropertyCategory,
          'AddressQualifier':AddressQualifier, 'IncidentType':IncidentType, 'Distance':Distance,
          'TotalOfPumpInLondon_Out':TotalOfPumpInLondon_Out, 'Station_Code_of_ressource':Station_Code_of_ressource,
          'IncidentStationGround_Code':IncidentStationGround_Code, 'PumpAvailable':PumpAvailable, 'month':month, 'temp':temp,
          'precip':precip, 'cloudcover':cloudcover, 'visibility':visibility, 'conditions':conditions, 'workingday':workingday,
          'school_holidays':school_holidays, 'congestion_rate':congestion_rate},name='x')
      st.session_state.df1=st.session_state.df1.append(new_row, ignore_index=True)
      st.session_state.count += 1
      st.write(st.session_state.count,' pump(s) sent')
  else:
      st.write (f'''<p style="color:Tomato;">You exceed the number of pumps to send</p>''',unsafe_allow_html=True)


if not st.session_state.df1.empty:
  if st.button('Predict Attendance Time for the pumps mobilised'):
      st.dataframe(st.session_state.df1.rename(index=lambda s: 'Camion n°'+ str(s+1)))
      nb_ligne=st.session_state.df1.shape[0]

      #--------------------------- TRAVAUX REGRESSION ---------------------------#
      # récupérer la table de détail des véhicules à envoyer
      df_reg = st.session_state.df1.rename(index=lambda s: 'Camion n°'+ str(s+1))
      #--------------------------- FIN TRAVAUX REGRESSION ---------------------------#

      with st.spinner('Working in progress ...'):
        time.sleep(3)
      df2=StandardSC_test(df,st.session_state.df1)
      df_temp=pd.concat([df,df2],ignore_index=True)
      dtemp=df_temp.tail().iloc[-nb_ligne:,:].reset_index().drop('index',axis=1)
      df_temp=Traitement2(df_temp)
      st.success('Done!')
      df_final=df_temp.tail().iloc[-nb_ligne:,:]
      dtemp=df_temp.tail().iloc[-nb_ligne:,:].reset_index().drop('index',axis=1)
      sgd_optuna = load(
          "data_outputs/ml_target_360/sgd_optuna.joblib"
          )
      y_pred=sgd_optuna.predict_proba(df_final)
      y_pred=pd.DataFrame(y_pred,columns=['% d\'arriver avant 360 secondes ',' % d\'arriver après 360 secondes'])
      y_pred['% d\'arriver avant 360 secondes ']=100*y_pred['% d\'arriver avant 360 secondes ']
      y_pred=y_pred.drop([' % d\'arriver après 360 secondes'],axis=1)
      y_pred=y_pred.rename(index=lambda s: 'Camion n°'+ str(s+1))
      
      #--------------------------- TRAVAUX REGRESSION ---------------------------#

      # Récupérer les paramètres de la régression
      reg_saved_files = 'streamlit/ml_attendance_time_saved/'
      num_var = ['Distance', 'TotalOfPumpInLondon_Out', 'temp', 'precip', 'cloudcover', 'visibility', 'congestion_rate']
      reg_scaler = load(reg_saved_files + 'reg_scaler.pkl')
      reg_df_columns = load(reg_saved_files + 'reg_df_columns.pkl')
      reg_df_columns_format = load(reg_saved_files + 'reg_df_columns_format.pkl')
      reg_data_dummies_columns = load(reg_saved_files + 'reg_data_dummies_columns.pkl')
      par = load(reg_saved_files + 'PassiveAggressiveRegressor.joblib')
      topslowest_code = list(pd.read_csv(reg_saved_files + 'reg_topslowest_code.csv')['0'])
      
      # COHERENCE VS FORMAT ENTRAINEMENT + PREPROCESSING
      # S'assurer que les colonnes sont dans le même ordre + format que le df principal
      df_reg = df_reg[reg_df_columns]
      df_reg = df_reg.astype(reg_df_columns_format)
      # Créer df au format dichotomisé
      df_reg_dum = pd.get_dummies(df_reg)
      # Normalisation des variables numériques
      df_reg_dum[num_var] = reg_scaler.transform(df_reg_dum[num_var])
      # Création d'un df vide, avec les colonnes de base du modèle
      df_reg_ml = pd.DataFrame(columns = reg_data_dummies_columns)
      # Reporter les valeurs de l'incident créé, dans le tableau au format du modèle
      df_reg_ml = df_reg_ml.append(df_reg_dum)
      # Remplacer les valeurs manquantes par 0 (ce sont les variables qui doivent être à zéro suite à dichotomisation)
      df_reg_ml = df_reg_ml.fillna(0)

      # PREDICTIONS
      # calculer la prédiction du modèle
      pred_df_reg = pd.DataFrame(np.round(par.predict(df_reg_ml), 0), columns = ['Temps estimé (secondes)'], index = df_reg_ml.index)
      # prévoir une colonne pour signaler les risques de sous-estimation légère
      pred_df_reg['Risque sous-estimation légère'] = ''
      # identifier le numéro (index) de la colonne d'alerte pour sous-estimation possible
      under_estim_col = pred_df_reg.columns.get_loc('Risque sous-estimation légère')
      # analyse des cas de sous-estimation possible selon critères définis
      for i in pred_df_reg.index:                      # pour chaque index du dataframe
          row_num = list(pred_df_reg.index).index(i)   # calculer le numéro (index) de la ligne
          # voir si le code ressource du véhicule est une caserne du top
          if df_reg['Station_Code_of_ressource'][i] in topslowest_code:
              pred_df_reg.iloc[row_num, under_estim_col] = 'oui'   # si c'est une caserne du top, signaler risque par "oui"
      # Afficher les prédictions
      pred_df_reg['Temps estimé (secondes)'] = pred_df_reg['Temps estimé (secondes)'].astype('int')
    #   pred_df_reg

      # Concaténer régression avec classification
      y_pred = y_pred.join(pred_df_reg)

      #--------------------------- FIN TRAVAUX REGRESSION ---------------------------#


    #   st.snow()
      st.write('Résultat: ',y_pred)

      #--------------------------- TRAVAUX GRAPHIQUE ---------------------------#
      
      # liste des couleurs
      graph_color = ['orange' if x > 360 else 'lime' for x in y_pred['Temps estimé (secondes)']]

      # paramètrage de la zone sur 3 colonnes (avec gestion proportion de taille pour rendu lisible)
      col_params = len(y_pred)
      col_center = 100 + col_params*20
      col_borders = 100 - col_params * 10
      col1, col2, col3 = st.columns([col_borders, col_center, col_borders])
      with col2:
          # Texte à afficher
          graph_text = []
          for i in y_pred.index:
              pred_seconds = y_pred['Temps estimé (secondes)'][i]
              pred_proba = int(round(y_pred['% d\'arriver avant 360 secondes '][i],0))
              
              if y_pred['Risque sous-estimation légère'][i] == 'oui':
                  graph_text.append("{}:{:02d} ++ | {}%"\
                                    .format(pred_seconds//60, pred_seconds%60, pred_proba))
              else:              
                  graph_text.append("{}:{:02d} | {}%"\
                                    .format(pred_seconds//60, pred_seconds%60, pred_proba))
          
          # Création du graphique
          import matplotlib.pyplot as plt
          fig = plt.figure(figsize= (len(y_pred)*2,4)) # *2 : à revoir en fonction du texte final retenu (plt.text)
          plt.title("\n\nEstimation du temps d'arrivée des véhicules envoyés\n\
          (min:sec  |  probabilité temps < 360 sec)\n\
          ('++' pour signaler risque de sous-estimation)\n")
          plt.ylabel('Temps en secondes')
          plt.axhline(360, c='red', ls='--', lw=0.6, label = '360 secondes')
          # Graphique en barres
          plt.bar(x = y_pred.index, height = y_pred["Temps estimé (secondes)"], color = graph_color)
          
          # Légende des couleurs utilisées
          import matplotlib.patches as mpatches
          lime_patch = mpatches.Patch(color='lime', label='<= 360 secondes')
          orange_patch = mpatches.Patch(color='orange', label='> 360 secondes')
          plt.legend(handles=[lime_patch, orange_patch], bbox_to_anchor =(0.5, -0.1), loc = 'upper center')
          
          # Afficher le texte
          for t in range(len(graph_text)):
              y_text = y_pred["Temps estimé (secondes)"][t]
              plt.text(x = y_pred.index[t],
                       y = y_text,
                       s = graph_text[t],
                       horizontalalignment = 'center',
                       verticalalignment = 'bottom');

          # Afficher le graphique
          st.pyplot(fig)
            
            
          st.write(df_final.shape)
  
                
               
        
  if st.button('New intervention ?', on_click=reset_button):
    st.session_state.df1=st.session_state.df1.iloc[0:0]
    df = pd.read_pickle(
        "data_outputs/base_ml.pkl"
        )
    st.session_state.count = 0
    st.session_state.count2 = 0
    st.experimental_rerun() 
