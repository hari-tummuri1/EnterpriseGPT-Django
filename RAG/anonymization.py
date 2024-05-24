from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
# Define the regex pattern in a Presidio `Pattern` object:
from presidio_analyzer import Pattern, PatternRecognizer


indian_phone_numbers_pattern = Pattern(
    name="indian_phone_numbers",
    regex=r"(?<!\w)(\(?(\+|00)?91\)?)?[ -]?\d{10}(?!\w)",
    score=1,
)

indian_phone_numbers_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER", patterns=[indian_phone_numbers_pattern]
)

aadhar_pattern = Pattern(
    name="aadhar_pattern",
    regex=r"(?<!\w)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\w)",
    score=1,
)

aadhar_recognizer = PatternRecognizer(
    supported_entity="AADHAR_NUMBER", patterns=[aadhar_pattern]
)

pan_pattern = Pattern(
    name="pan_pattern",
    regex=r"(?<!\w)[A-Z]{5}[0-9]{4}[A-Z](?!\w)",
    score=1,
)

pan_recognizer = PatternRecognizer(
    supported_entity="PAN_NUMBER", patterns=[pan_pattern]
)

# Driving license number pattern
driving_license_pattern = Pattern(
    name="driving_license_pattern",
    regex=r"(?<!\w)[A-Z]{2}[ -]?[0-9]{2,3}[ -]?[0-9]{4,7}(?!\w)",
    score=1,
)

driving_license_recognizer = PatternRecognizer(
    supported_entity="DRIVING_LICENSE_NUMBER", patterns=[driving_license_pattern]
)

anonymizer = PresidioReversibleAnonymizer()


anonymizer.add_recognizer(aadhar_recognizer)
anonymizer.add_recognizer(driving_license_recognizer)
anonymizer.add_recognizer(pan_recognizer)
anonymizer.add_recognizer(indian_phone_numbers_recognizer)
# anonymizer.deanonymizer_mapping


def anonymize(text):
    anonymized_text = anonymizer.anonymize(text)
    anonymizer.save_deanonymizer_mapping("deanonymizer_mapping.json")
    return anonymized_text

def deAnonymize(text):
    anonymizer.load_deanonymizer_mapping("deanonymizer_mapping.json")
    original_text = anonymizer.deanonymize(text)
    return original_text

# text = "My name is Slim Shady, call me at 313-666-7440 or email me at real.slim.shady@gmail.com. By the way, my card number is: 98487654634253"
# text = "I need help resetting my password on a website. My name is John Smith. Can you assist me?"
# ann_text = anonymize(text)
# print(ann_text)
# print(deAnonymize("Thank you for reaching out, John. To help you reset your password, could you please confirm your email address and the website you are trying to access? This will allow me to assist you in a safe and secure manner."))
