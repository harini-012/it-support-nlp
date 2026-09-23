from nlp import NLPPreprocessor


processor = NLPPreprocessor()


text = """
My laptop cannot connect to the company VPN
and the internet connection keeps failing.
"""


result = processor.clean_text(text)


print("Original:")
print(text)

print("\nProcessed:")
print(result)