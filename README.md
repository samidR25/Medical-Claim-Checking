Medical Claim Verification System for COVID-19
This project provides a system for automatically verifying medical claims related to COVID-19. It consists of two main components:

Counter-Claim Generation: A module for generating counter-claims to test the robustness of verification models
Claim Verification: A module for verifying the truthfulness of COVID-19 related claims

Project Structure
medical-claim-checker/
├── counter-claim-generation/  # Counter-claim generation component
│   ├── data/                  # Data files including dictionaries
│   ├── src/                   # Source code for counter-claim generation
│   ├── enhance_covidfact_dataset.py  # Script to enhance dataset with counter-claims
│   └── evaluate_enhancement.py       # Evaluation script
├── claim-verification/        # Claim verification component
│   ├── data/                  # Dataset files
│   ├── notebooks/             # Jupyter notebooks with experiments
│   └── src/                   # Source code for claim verification
└── requirements.txt           # Project dependencies
Installation


Create and activate a virtual environment:
bashpython -m venv covid_env
source covid_env/bin/activate  # On Windows: covid_env\Scripts\activate

Install dependencies:
bashpip install -r requirements.txt
python -m spacy download en_core_web_sm


Counter-Claim Generation
Enhancing Dataset with Counter-Claims

Navigate to the counter-claim generation directory:
bashcd counter-claim-generation

Run the enhancement script:
bashpython enhance_covidfact_dataset.py

Evaluate the enhanced dataset:
bashpython evaluate_enhancement.py


Using the Counter-Claim Generator
pythonfrom src.counter_claim_generation.improved_generator import ImprovedCounterClaimGenerator

# Initialize the generator
generator = ImprovedCounterClaimGenerator(
    use_srl=True,
    use_entity_sub=True,
    use_quantity_mod=True,
    use_span_replacement=True,
    use_dependency_structure=True,
    contradiction_threshold=0.7
)

# Generate counter-claims
claim = "Masks are effective in preventing COVID-19 transmission"
counter_claims = generator.generate_counter_claims(claim, num_candidates=3)

for cc in counter_claims:
    print(f"Original: {claim}")
    print(f"Counter-claim: {cc['counter_claim']}")
    print(f"Strategy: {cc['strategy']}")
    print(f"Contradiction score: {cc['contradiction_score']:.2f}")
    print("-" * 50)
Claim Verification
Running the Verification Model

Navigate to the claim verification directory:
claim-verification

Open the Jupyter notebook to explore the claim verification model:
bashjupyter notebook notebooks/COVID-19_Claim_Verification.ipynb


Using the Verification Model
pythonfrom src.claim_verification.verifier import MedicalClaimVerifier

# Initialize the verifier
verifier = MedicalClaimVerifier()

# Load a pre-trained model
verifier = MedicalClaimVerifier.load("models/biobert_covid_claims")

# Verify a claim
results = verifier.predict(["Masks are effective in preventing COVID-19 transmission"])
print(results["readable_predictions"])  # ['SUPPORTED'] or ['REFUTED']
Datasets

COVIDFACT_dataset.jsonl: Original COVID fact-checking dataset
enhanced_full_dataset.jsonl: Dataset enhanced with automatically generated counter-claims
filtered_dataset.jsonl: Filtered version of the dataset used for training

Models
The system uses transformer-based models for claim verification:

BioBERT: A biomedical language representation model for claim verification
RoBERTa-large-MNLI: For contradiction detection in counter-claim generation

Notes
Note: When running the claim verification model for the first time, you can choose to download all the pretrained model files (approximately 1.5GB) as they will be automatically downloaded from the Hugging Face Model Hub. The enhanced dataset can be regenerated using the enhance_covidfact_dataset.py script in the counter-claim-generation directory.

Counter-claim generation requires significant computational resources
Using a GPU will significantly speed up the inference process

Acknowledgements
This project uses the COVIDFACT dataset from the paper "COVID-Fact: Fact Extraction and Verification of Real-World Claims on COVID-19 Pandemic" by Saakyan et al.
