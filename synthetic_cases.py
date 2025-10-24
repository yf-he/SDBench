"""Synthetic NEJM-style cases for SDBench testing."""

from data_models import CaseFile

def create_synthetic_case_1() -> CaseFile:
    """Create a synthetic NEJM-style case: Histoplasmosis with mediastinal lymphadenopathy."""
    return CaseFile(
        case_id="SYNTH_001",
        initial_abstract="A 34-year-old woman presents with a 6-week history of progressive dyspnea, dry cough, and fatigue. She reports a 10-pound weight loss over the past month and night sweats. Physical examination reveals bilateral cervical lymphadenopathy and decreased breath sounds at the right lung base.",
        full_case_text="""
        PRESENTATION OF CASE
        
        A 34-year-old woman was admitted to the hospital because of progressive dyspnea, dry cough, and fatigue.
        
        The patient had been in her usual state of health until 6 weeks before admission, when she began to experience progressive dyspnea on exertion, initially occurring only with climbing stairs but gradually worsening to the point that she became short of breath with minimal activity. She also developed a dry, nonproductive cough that was worse at night and was associated with occasional chest tightness. She reported a 10-pound weight loss over the past month and drenching night sweats that required changing her clothes. She denied fever, chills, hemoptysis, or chest pain. She had no recent travel history but reported that she had been cleaning out her basement 2 months before the onset of symptoms, where she had found evidence of bird droppings.
        
        The patient was a nonsmoker and had no known allergies. Her medical history was notable only for mild asthma, which had been well controlled with an albuterol inhaler as needed. She worked as an office manager and lived in a suburban area in the Midwestern United States. She had no pets and no recent sick contacts.
        
        On physical examination, the patient appeared thin and mildly ill. Her temperature was 37.2°C (99.0°F), blood pressure 110/70 mm Hg, pulse 95 beats per minute, respiratory rate 20 breaths per minute, and oxygen saturation 94% while breathing ambient air. Her height was 165 cm and weight 58 kg (body-mass index, 21.3). The examination of the head and neck revealed bilateral cervical lymphadenopathy, with nodes that were firm, nontender, and measuring up to 2 cm in diameter. The examination of the chest revealed decreased breath sounds at the right lung base, with no wheezing or crackles. The cardiac examination was normal. The abdomen was soft and nontender, with no organomegaly. The extremities showed no clubbing, cyanosis, or edema.
        
        Laboratory studies revealed the following values: hemoglobin, 10.2 g per deciliter (reference range, 12.0 to 15.5); hematocrit, 31.2% (reference range, 36.0 to 46.0); white-cell count, 12,400 per cubic millimeter (reference range, 4500 to 11,000), with 78% neutrophils, 15% lymphocytes, 5% monocytes, and 2% eosinophils; platelet count, 425,000 per cubic millimeter (reference range, 150,000 to 450,000); and erythrocyte sedimentation rate, 45 mm per hour (reference range, 0 to 20). The serum chemistry panel was normal except for a total protein level of 8.2 g per deciliter (reference range, 6.0 to 8.0) and an albumin level of 3.1 g per deciliter (reference range, 3.5 to 5.0). The serum lactate dehydrogenase level was 280 U per liter (reference range, 140 to 280). The results of liver-function tests were normal.
        
        A chest radiograph showed bilateral hilar lymphadenopathy and a small right pleural effusion. A computed tomographic (CT) scan of the chest with intravenous contrast material revealed extensive mediastinal and hilar lymphadenopathy, with the largest nodes measuring up to 3 cm in diameter. There were also small bilateral pleural effusions and subtle bilateral lower-lobe infiltrates. No pulmonary nodules or masses were identified.
        
        A tuberculin skin test was negative. The results of serologic testing for human immunodeficiency virus (HIV) were negative. The results of serologic testing for histoplasmosis were positive, with a complement-fixation titer of 1:32 and an immunodiffusion test that was positive for H and M bands. The results of serologic testing for coccidioidomycosis and blastomycosis were negative.
        
        A bronchoscopy with bronchoalveolar lavage was performed. The lavage fluid contained 85% lymphocytes, 10% neutrophils, and 5% macrophages. The results of cytologic examination were negative for malignant cells. The results of bacterial, fungal, and mycobacterial cultures were negative. The results of polymerase-chain-reaction (PCR) testing for Mycobacterium tuberculosis were negative.
        
        A mediastinoscopy with biopsy of a mediastinal lymph node was performed. Histologic examination of the biopsy specimen revealed noncaseating granulomas with multinucleated giant cells. Special stains for acid-fast bacilli and fungi were negative. The results of culture of the biopsy specimen were negative for bacteria, fungi, and mycobacteria.
        
        The patient was treated with itraconazole, 200 mg twice daily, for 6 months. At follow-up 3 months after the completion of therapy, her symptoms had resolved, and a repeat chest CT scan showed complete resolution of the mediastinal lymphadenopathy and pleural effusions.
        
        DISCUSSION
        
        This case illustrates the presentation of chronic pulmonary histoplasmosis, a form of histoplasmosis that occurs in patients with normal immune function. The disease is caused by Histoplasma capsulatum, a dimorphic fungus that is endemic to the central and eastern United States, particularly in areas around the Ohio and Mississippi River valleys. The organism is found in soil contaminated with bird or bat droppings, and infection typically occurs through inhalation of conidia.
        
        The clinical presentation of chronic pulmonary histoplasmosis is often insidious, with symptoms developing over weeks to months. Common symptoms include progressive dyspnea, dry cough, weight loss, night sweats, and fatigue. Physical examination may reveal lymphadenopathy, particularly cervical lymphadenopathy, and signs of pleural effusion or consolidation.
        
        The diagnosis of histoplasmosis is typically made on the basis of a combination of clinical presentation, radiographic findings, and serologic testing. Serologic testing for histoplasmosis includes complement-fixation and immunodiffusion tests. A complement-fixation titer of 1:32 or higher is considered positive, and the presence of H and M bands on immunodiffusion testing is highly specific for histoplasmosis.
        
        Radiographic findings in chronic pulmonary histoplasmosis typically include mediastinal and hilar lymphadenopathy, often with associated pleural effusions and lower-lobe infiltrates. The lymphadenopathy may be extensive and can sometimes be mistaken for lymphoma or other malignancies.
        
        The treatment of chronic pulmonary histoplasmosis typically involves antifungal therapy with itraconazole for 6 to 12 months. The prognosis is generally good with appropriate treatment, and most patients experience complete resolution of symptoms and radiographic abnormalities.
        
        This case highlights the importance of considering endemic fungal infections in the differential diagnosis of mediastinal lymphadenopathy, particularly in patients who live in or have traveled to endemic areas. The diagnosis can be challenging, as the clinical presentation and radiographic findings may mimic other conditions, including malignancy and tuberculosis.
        """,
        ground_truth_diagnosis="Chronic pulmonary histoplasmosis with mediastinal lymphadenopathy",
        publication_year=2024,
        is_test_case=True
    )

def create_synthetic_case_2() -> CaseFile:
    """Create a synthetic NEJM-style case: Autoimmune hemolytic anemia."""
    return CaseFile(
        case_id="SYNTH_002",
        initial_abstract="A 28-year-old woman presents with a 2-week history of fatigue, jaundice, and dark urine. She reports no recent illness or medication use. Physical examination reveals scleral icterus, pallor, and mild splenomegaly. Laboratory studies show anemia with evidence of hemolysis.",
        full_case_text="""
        PRESENTATION OF CASE
        
        A 28-year-old woman was admitted to the hospital because of fatigue, jaundice, and dark urine.
        
        The patient had been in her usual state of health until 2 weeks before admission, when she began to experience progressive fatigue and weakness. She also noticed that her skin and eyes had become yellow, and her urine had become dark brown. She denied fever, chills, abdominal pain, or recent illness. She had not taken any new medications and had no known drug allergies. She had no recent travel history and no sick contacts.
        
        The patient was a nonsmoker and had no significant medical history. She worked as a teacher and lived in a suburban area. She had no pets and no family history of blood disorders.
        
        On physical examination, the patient appeared pale and mildly icteric. Her temperature was 36.8°C (98.2°F), blood pressure 105/65 mm Hg, pulse 88 beats per minute, respiratory rate 16 breaths per minute, and oxygen saturation 98% while breathing ambient air. Her height was 170 cm and weight 62 kg (body-mass index, 21.5). The examination of the head and neck revealed scleral icterus. The examination of the chest was normal. The cardiac examination revealed a grade 2/6 systolic murmur at the left sternal border. The abdomen was soft and nontender, with the spleen palpable 2 cm below the left costal margin. The extremities showed no clubbing, cyanosis, or edema.
        
        Laboratory studies revealed the following values: hemoglobin, 7.8 g per deciliter (reference range, 12.0 to 15.5); hematocrit, 23.4% (reference range, 36.0 to 46.0); white-cell count, 8,200 per cubic millimeter (reference range, 4500 to 11,000), with 65% neutrophils, 28% lymphocytes, 5% monocytes, and 2% eosinophils; platelet count, 180,000 per cubic millimeter (reference range, 150,000 to 450,000); and reticulocyte count, 12% (reference range, 0.5 to 2.5). The peripheral blood smear showed spherocytes, polychromasia, and nucleated red blood cells. The serum chemistry panel revealed the following values: total bilirubin, 4.2 mg per deciliter (reference range, 0.3 to 1.2), with a direct bilirubin of 0.8 mg per deciliter; lactate dehydrogenase, 450 U per liter (reference range, 140 to 280); and haptoglobin, less than 10 mg per deciliter (reference range, 30 to 200). The results of liver-function tests were normal.
        
        A direct antiglobulin test (Coombs test) was positive, with IgG and C3d coating the red blood cells. The results of serologic testing for autoimmune diseases, including antinuclear antibody, rheumatoid factor, and anti-double-stranded DNA antibody, were negative. The results of serologic testing for viral infections, including hepatitis B and C, human immunodeficiency virus, and Epstein-Barr virus, were negative.
        
        A bone marrow biopsy was performed, which showed erythroid hyperplasia with a myeloid-to-erythroid ratio of 1:2. The results of cytogenetic analysis were normal.
        
        The patient was treated with prednisone, 1 mg per kilogram of body weight per day, and her symptoms began to improve within 1 week. At follow-up 1 month after the initiation of therapy, her hemoglobin level had increased to 11.2 g per deciliter, and her bilirubin level had normalized. The prednisone dose was gradually tapered over 6 months, and the patient remained in remission.
        
        DISCUSSION
        
        This case illustrates the presentation of autoimmune hemolytic anemia (AIHA), a condition in which the immune system produces antibodies that attack and destroy red blood cells. AIHA can be classified as warm or cold, depending on the temperature at which the antibodies are most active. Warm AIHA, which is more common, is typically caused by IgG antibodies that are most active at body temperature.
        
        The clinical presentation of AIHA is often insidious, with symptoms developing over days to weeks. Common symptoms include fatigue, weakness, jaundice, and dark urine. Physical examination may reveal pallor, scleral icterus, and splenomegaly. The severity of symptoms depends on the rate of hemolysis and the patient's ability to compensate with increased erythropoiesis.
        
        The diagnosis of AIHA is typically made on the basis of a combination of clinical presentation, laboratory findings, and the results of the direct antiglobulin test. Laboratory findings consistent with hemolysis include anemia, elevated reticulocyte count, elevated lactate dehydrogenase level, elevated indirect bilirubin level, and decreased haptoglobin level. The presence of spherocytes on peripheral blood smear is also suggestive of hemolysis.
        
        The direct antiglobulin test is the cornerstone of the diagnosis of AIHA. A positive test indicates that antibodies or complement components are coating the red blood cells. In warm AIHA, the test is typically positive for IgG, with or without C3d.
        
        The treatment of AIHA typically involves immunosuppressive therapy, with corticosteroids being the first-line treatment. Most patients respond to prednisone, with improvement in symptoms and laboratory values within 1 to 2 weeks. The dose is gradually tapered over several months, and many patients can be maintained in remission with low-dose corticosteroids or other immunosuppressive agents.
        
        The prognosis of AIHA is generally good with appropriate treatment, although some patients may experience relapses. The disease can be associated with other autoimmune conditions, and patients should be monitored for the development of additional autoimmune disorders.
        
        This case highlights the importance of considering AIHA in the differential diagnosis of anemia, particularly in patients with evidence of hemolysis. The diagnosis can be challenging, as the clinical presentation and laboratory findings may mimic other conditions, including other causes of hemolytic anemia.
        """,
        ground_truth_diagnosis="Autoimmune hemolytic anemia (warm type)",
        publication_year=2024,
        is_test_case=True
    )

def create_synthetic_case_3() -> CaseFile:
    """Create a synthetic NEJM-style case: Pheochromocytoma."""
    return CaseFile(
        case_id="SYNTH_003",
        initial_abstract="A 45-year-old man presents with episodes of severe headaches, palpitations, and diaphoresis lasting 10-15 minutes. The episodes occur 2-3 times per week and are often triggered by stress or physical activity. Physical examination reveals hypertension and tachycardia during an episode.",
        full_case_text="""
        PRESENTATION OF CASE
        
        A 45-year-old man was referred to the endocrinology clinic because of episodes of severe headaches, palpitations, and diaphoresis.
        
        The patient had been in his usual state of health until 3 months before presentation, when he began to experience episodes of severe, pounding headaches that were often associated with palpitations and profuse sweating. The episodes typically lasted 10 to 15 minutes and occurred 2 to 3 times per week. They were often triggered by stress, physical activity, or changes in position. He also reported episodes of anxiety and tremors during these attacks. He denied chest pain, shortness of breath, or loss of consciousness.
        
        The patient was a nonsmoker and had no known drug allergies. His medical history was notable only for mild hypertension, which had been diagnosed 2 years earlier and was well controlled with lisinopril, 10 mg daily. He worked as a construction manager and lived in a suburban area. He had no family history of endocrine disorders or sudden cardiac death.
        
        On physical examination, the patient appeared anxious and diaphoretic. His temperature was 36.9°C (98.4°F), blood pressure 180/110 mm Hg, pulse 110 beats per minute, respiratory rate 18 breaths per minute, and oxygen saturation 98% while breathing ambient air. His height was 178 cm and weight 85 kg (body-mass index, 26.8). The examination of the head and neck was normal. The examination of the chest was normal. The cardiac examination revealed a regular rhythm with no murmurs. The abdomen was soft and nontender, with no organomegaly. The extremities showed no clubbing, cyanosis, or edema.
        
        Laboratory studies revealed the following values: hemoglobin, 14.2 g per deciliter (reference range, 13.8 to 17.2); hematocrit, 42.1% (reference range, 40.7 to 50.3); white-cell count, 7,800 per cubic millimeter (reference range, 4500 to 11,000); platelet count, 285,000 per cubic millimeter (reference range, 150,000 to 450,000); and serum glucose, 110 mg per deciliter (reference range, 70 to 100). The serum chemistry panel was normal. The results of thyroid-function tests were normal.
        
        A 24-hour urine collection for metanephrines and normetanephrines revealed the following values: metanephrines, 1,200 μg per 24 hours (reference range, 0 to 400); normetanephrines, 2,800 μg per 24 hours (reference range, 0 to 900); and total metanephrines, 4,000 μg per 24 hours (reference range, 0 to 1,300). The plasma metanephrines were also elevated: metanephrines, 0.8 nmol per liter (reference range, 0 to 0.5); and normetanephrines, 2.1 nmol per liter (reference range, 0 to 0.9).
        
        A computed tomographic (CT) scan of the abdomen with intravenous contrast material revealed a 4.5-cm mass in the left adrenal gland. The mass was well-circumscribed and had a heterogeneous appearance with areas of necrosis. No other abnormalities were identified.
        
        A magnetic resonance imaging (MRI) scan of the abdomen confirmed the presence of a left adrenal mass with characteristics consistent with a pheochromocytoma. The mass showed high signal intensity on T2-weighted images and heterogeneous enhancement after the administration of gadolinium contrast material.
        
        The patient was treated with phenoxybenzamine, 10 mg twice daily, which was gradually increased to 20 mg twice daily over 2 weeks. His blood pressure improved, and the episodes of headaches and palpitations decreased in frequency and severity. After 3 weeks of alpha-blockade, he underwent laparoscopic left adrenalectomy.
        
        The surgical specimen was a 4.2-cm adrenal mass weighing 45 g. Histologic examination revealed a pheochromocytoma with no evidence of capsular invasion or vascular invasion. The results of immunohistochemical staining were positive for chromogranin A and synaptophysin. The Ki-67 proliferation index was less than 5%.
        
        Postoperatively, the patient's blood pressure normalized, and he had no further episodes of headaches or palpitations. At follow-up 6 months after surgery, his blood pressure remained normal without antihypertensive medications, and his plasma metanephrines were within normal limits.
        
        DISCUSSION
        
        This case illustrates the presentation of a pheochromocytoma, a rare neuroendocrine tumor that arises from chromaffin cells of the adrenal medulla. Pheochromocytomas are part of the paraganglioma family of tumors and can occur sporadically or as part of hereditary syndromes, such as multiple endocrine neoplasia type 2, von Hippel-Lindau disease, and neurofibromatosis type 1.
        
        The clinical presentation of pheochromocytoma is often characterized by the classic triad of headaches, palpitations, and diaphoresis, although not all patients present with all three symptoms. The episodes, often referred to as "spells," can be triggered by various factors, including stress, physical activity, changes in position, and certain medications. The severity and frequency of episodes can vary widely among patients.
        
        The diagnosis of pheochromocytoma is typically made on the basis of a combination of clinical presentation, biochemical testing, and imaging studies. Biochemical testing involves the measurement of plasma or urinary metanephrines and normetanephrines, which are the metabolites of epinephrine and norepinephrine, respectively. Elevated levels of these metabolites are highly sensitive and specific for the diagnosis of pheochromocytoma.
        
        Imaging studies are used to localize the tumor and assess its characteristics. CT and MRI are the most commonly used imaging modalities. Pheochromocytomas typically appear as well-circumscribed adrenal masses with heterogeneous enhancement. On MRI, they often show high signal intensity on T2-weighted images, which is referred to as the "light bulb" sign.
        
        The treatment of pheochromocytoma involves surgical resection, typically through laparoscopic adrenalectomy. Before surgery, patients should be treated with alpha-adrenergic blockade to prevent hypertensive crises during manipulation of the tumor. Phenoxybenzamine, a nonselective alpha-adrenergic antagonist, is commonly used for this purpose. Beta-adrenergic blockade may be added if there are persistent tachycardia or arrhythmias, but it should only be started after adequate alpha-blockade to prevent unopposed alpha-adrenergic stimulation.
        
        The prognosis of pheochromocytoma is generally good with appropriate treatment, and most patients experience complete resolution of symptoms after surgical resection. However, the disease can recur, and patients should be monitored for the development of additional tumors, particularly in cases of hereditary syndromes.
        
        This case highlights the importance of considering pheochromocytoma in the differential diagnosis of episodic hypertension and adrenergic symptoms. The diagnosis can be challenging, as the clinical presentation may mimic other conditions, including anxiety disorders and other causes of episodic hypertension.
        """,
        ground_truth_diagnosis="Pheochromocytoma of the left adrenal gland",
        publication_year=2024,
        is_test_case=True
    )

def get_all_synthetic_cases() -> list[CaseFile]:
    """Return all synthetic cases."""
    return [
        create_synthetic_case_1(),
        create_synthetic_case_2(),
        create_synthetic_case_3()
    ]
