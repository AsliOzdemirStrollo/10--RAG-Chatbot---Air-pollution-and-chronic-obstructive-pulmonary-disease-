# evaluation/evaluation_questions.py

EVALUATION_DATA = [
    {
        "question": "What are the main short-term and long-term health effects of air pollution on patients with COPD?",
        "ground_truth": (
            "According to the document, air pollution causes both short-term and long-term adverse effects "
            "in patients with COPD. These effects include exacerbation of existing symptoms, impaired lung "
            "function, and increased hospitalization and mortality rates. Long-term exposure to high "
            "concentrations of pollutants may also increase the incidence of COPD."
        )
    },
    {
        "question": "Which air pollutants are most commonly monitored, and how do PM2.5 and PM10 differ in their effects on the respiratory system?",
        "ground_truth": (
            "The most commonly monitored air pollutants are particulate matter (PM) and gaseous pollutants such "
            "as SO2, NO2, CO, and O3. PM includes PM10 and fine PM2.5. PM10 mainly accumulates in the upper "
            "respiratory tract, while PM2.5 penetrates both the upper and lower respiratory tracts, especially the "
            "small airways and alveoli, causing more significant respiratory harm."
        )
    },
    {
        "question": "What epidemiological evidence does the document provide about the relationship between air pollution and COPD prevalence?",
        "ground_truth": (
            "The document reports mixed epidemiological findings. Some large cohort studies, such as ESCAPE, "
            "found no consistent association between long-term exposure to PM or NO2 and COPD prevalence, "
            "while other studies showed significant associations. For example, a UK Biobank analysis found that "
            "PM2.5, PM10, and NO2 levels were significantly associated with increased COPD prevalence, with "
            "odds ratios of 1.52, 1.08, and 1.12 respectively. In China, similar cross-sectional findings were "
            "reported in 2017."
        )
    },
    {
    "question": "What overall relationship does the document describe between air pollution and COPD mortality?",
    "ground_truth": (
        "The document explains that higher levels of air pollutants are associated with increased COPD-related "
        "mortality. Pollutants such as PM2.5, PM10, NO2, SO2, and O3 all contribute to a greater risk of death "
        "in individuals with COPD."
    )
    },
    {
        "question": "What mechanisms are proposed in the document to explain how air pollution contributes to the development or worsening of COPD?",
        "ground_truth": (
            "The document describes several possible mechanisms, including inflammatory damage, oxidative "
            "stress, and genetic or epigenetic damage. PM2.5 can induce inflammatory cell infiltration and "
            "stimulate the release of cytokines such as IL-6 and TNF-alpha. Particulate matter also generates "
            "reactive oxygen species, leading to oxidative injury. Additionally, air pollutants can cause DNA "
            "damage and alter gene expression through changes in DNA methylation and other epigenetic pathways."
        )
    },
    {
        "question": "What protective strategies against air pollution exposure are described in the document for patients with COPD?",
        "ground_truth": (
            "The document describes several protective strategies, including government policies such as China's "
            "Air Pollution Prevention and Control Action Plan and the Clean Air Act in the United States. Group "
            "interventions include reducing indoor air pollution by using clean fuels, improved cookstoves, and "
            "better kitchen ventilation. Individual measures include limiting outdoor activity during heavy pollution, "
            "wearing masks such as N95 respirators, and using indoor air purifiers. Some medical interventions, such "
            "as nebulization therapies, may also help eliminate inhaled particulate matter from the respiratory tract."
        )
    }
]