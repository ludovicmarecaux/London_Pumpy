# London_Pumpy

La brigade des pompiers de Londres est le service d'incendie et de sauvetage le plus actif du Royaume-Uni et l'une des plus grandes organisations de lutte contre l'incendie et de sauvetage au monde.

Pour la période 2017-2021, la 6ème version du “London Safety Plan” (document rassemblant les missions et objectifs de la brigrade des pompiers de Londres, source : https://www.london-fire.gov.uk/about-us/london-safety-plan/) était en vigueur, définissant les priorités afin notamment de réduire le temps d’arrivée sur site des camions pour rendre la ville de Londres plus sûre. Un des objectifs fixés était l’arrivée sur chaque lieu d’incident en moins de 360s.

A partir de l’année 2022 et pour les 5 prochaines années, un nouveau plan d’action, London Pump.Py est mis en place. L’objectif principal sera d’analyser et prédire le temps d’intervention et de mobilisation de la brigade des pompiers de Londres.

Pour ce travail, nous possédons deux tables composées de jeux de données répertoriant toutes les informations des interventions entre Janvier 2018 et Octobre 2021 :
les informations sur l'appareil mobilisé, son lieu de déploiement et les heures d'arrivée sur les lieux de l'incident et 
les informations sur la date et le lieu de l'incident ainsi que sur le type d'incident traité.

Pour cela, nous mettrons en place trois modélisations. Dans un premier temps, nous créerons un modèle capable de définir le nombre de camions à envoyer en fonction des éléments connus au moment de l’appel. Ensuite, nous prédirons si les véhicules envoyés sur l’intervention arriveront dans le délai objectif des 360s. Et enfin, nous affinerons notre étude en estimant le temps d’arrivée sur site des premiers secours. 

L'application issu de ce travail est réalisé sur Streamlit : https://ludovicmarecaux-london-pumpy-app-d7h8wh.streamlitapp.com/


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The London Fire Brigade is the busiest fire and rescue service in the UK and one of the largest firefighting and rescue organizations in the world.

For the period 2017-2021, the 6th version of the "London Safety Plan" (document bringing together the missions and objectives of the London fire brigade, source: https://www.london-fire.gov.uk/about-us /london-safety-plan/) was in force, defining priorities in order in particular to reduce the time of arrival on site of trucks to make the city of London safer. One of the objectives set was to arrive at each incident site in less than 360s.

From the year 2022 and for the next 5 years, a new action plan, London Pump.Py is implemented. The main objective will be to analyze and predict the intervention and mobilization time of the London fire brigade.

For this work, we have two tables made up of data sets listing all the information from the interventions between January 2018 and October 2021:
information on the mobilized aircraft, its place of deployment and the times of arrival at the scene of the incident and
information on the date and place of the incident as well as on the type of incident handled.

For this, we will set up three models. First, we will create a model capable of defining the number of trucks to send based on what is known at the time of the call. Then, we will predict if the vehicles sent to the intervention will arrive within the objective time of 360s. And finally, we will refine our study by estimating the time of arrival on site of first aid.

The application resulting from this work is made on Streamlit: https://ludovicmarecaux-london-pumpy-app-d7h8wh.streamlitapp.com/
