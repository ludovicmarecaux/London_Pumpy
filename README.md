# London_Pumpy

La brigade des pompiers de Londres est le service d'incendie et de sauvetage le plus actif du Royaume-Uni et l'une des plus grandes organisations de lutte contre l'incendie et de sauvetage au monde.

Pour la période 2017-2021, la 6ème version du “London Safety Plan” (document rassemblant les missions et objectifs de la brigrade des pompiers de Londres, source : https://www.london-fire.gov.uk/about-us/london-safety-plan/) était en vigueur, définissant les priorités afin notamment de réduire le temps d’arrivée sur site des camions pour rendre la ville de Londres plus sûre. Un des objectifs fixés était l’arrivée sur chaque lieu d’incident en moins de 360s.

A partir de l’année 2022 et pour les 5 prochaines années, un nouveau plan d’action, London Pump.Py est mis en place. L’objectif principal sera d’analyser et prédire le temps d’intervention et de mobilisation de la brigade des pompiers de Londres.

Pour cela, nous mettrons en place trois modélisations. Dans un premier temps, nous créerons un modèle capable de définir le nombre de camions à envoyer en fonction des éléments connus au moment de l’appel. Ensuite, nous prédirons si les véhicules envoyés sur l’intervention arriveront dans le délai objectif des 360s. Et enfin, nous affinerons notre étude en estimant le temps d’arrivée sur site des premiers secours. 

Pour ce travail, nous possédons deux tables composées de jeux de données répertoriant toutes les informations des interventions entre Janvier 2018 et Octobre 2021 :
les informations sur l'appareil mobilisé, son lieu de déploiement et les heures d'arrivée sur les lieux de l'incident et 
les informations sur la date et le lieu de l'incident ainsi que sur le type d'incident traité
