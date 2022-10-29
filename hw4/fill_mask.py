"""
Starter Code for CS475 HW4: Ethnic Bias in Language Models.
!!! Warning !!! Some model outputs may be offensive or upsetting.

Below can also be found in Colab: https://colab.research.google.com/drive/1BWluZr3Kh4lHxFjOfX9g1Xjl96c9wYgL?usp=sharing
"""
import math
from pprint import pprint
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

import nltk
nltk.download('omw-1.4')

LANG = "EN"  # "ES"

if LANG == "EN":
  from pattern.en import pluralize
elif LANG == "ES":
  from pattern.es import pluralize

# You can use model checkpoint other than "bert-base-cased"
# See: https://huggingface.co/models for the available models that you can easily use.

if LANG == "EN":
  tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
  model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
elif LANG == "ES":
  tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
  model = AutoModelForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

# You can try out different targets, attribute, and/or template for your exploration

# Countries from the original paper.

# targets = [
#   "America", "Canada", "Japan", "China", "Korea", "England", "France",
#   "Germany", "Mexico", "Iraq", "Ireland", "Iran", "Saudi", "Russia", "Vietnam",
#   "Thailand", "Australia", "Spain", "Turkey", "Israel", "Italy", "Egypt", "Somalia",
#   "India", "Brazil", "Colombia", "Greece", "Afghanistan", "Cuba", "Syria"
# ]
# targets = [
#   "América", "Canadá", "Japón", "China", "Corea", "Inglaterra", "Francia",
#   "Alemania", "México", "Iraq", "Irlanda", "Irán", "Arabia Saudita", "Rusia", "Vietnam",
#   "Tailandia", "Australia", "España", "Turquía", "Israel", "Italia", "Egipto", "Somalia",
#   "India", "Brasil", "Colombia", "Grecia", "Afganistán", "Cuba", "Siria"
# ]

# Countries for our experiment (N countries for each religion)
# Christianity, Muslim, Hinduism, Buddism

if LANG == "EN":
  targets = [
    "Korea", "Japan", "India", "Bangladesh", "Indonesia", "America", 
    "Israel", "Mexico", "Brazil", "Greece", "Colombia", "Russia", "Philippines", 
    "Paraguay", "Nigeria", "China", "Congo", "Germany", "Ethiopia", "Iran", 
    "Iraq", "Syria", "Turkey", "Afghanistan", "Egypt", "Algeria", "Pakistan", 
    "Qatar", "Morocco", "Saudi", "Somalia", "Nepal", 
    "Pakistan", "Malaysia", "Myanmar", "Cambodia", "Mongolia", 
    "China", "Thailand", "Singapore"
  ]
elif LANG == "ES":
  targets = [
    "Corea", "Japón", "India", "Bangladesh", "Indonesia", "América",
    "Israel", "México", "Brasil", "Grecia", "Colombia", "Rusia", "Filipinas",
    "Paraguay", "Nigeria", "China", "Congo", "Alemania", "Etiopía", "Irán",
    "Irak", "Siria", "Turquía", "Afganistán", "Egipto", "Argelia", "Pakistán",
    "Qatar", "Marruecos", "Saudita", "Somalia", "Nepal",
    "Pakistán", "Malasia", "Myanmar", "Camboya", "Mongolia",
    "China", "Tailandia", "Singapur"
  ]
# "Sri Lanka", "Gambia", "Azerbaizán", "Bután", "Laos", 

# Attributes
attr_type = "religion"  # "occ", "religion"
if LANG == "EN":
  with open(f'{attr_type}_en.txt', 'r') as f:
    attributes = f.readlines()
elif LANG == "ES":
  with open(f'{attr_type}_es.txt', 'r') as f:
    attributes = f.readlines()

# Templates
if LANG == "EN":
  with open('templates_en.txt', 'r') as f:
    templates = f.readlines()
elif LANG == "ES":
  with open('templates_es.txt', 'r') as f:
    templates = f.readlines()


# For more documentation of fill-mask pipeline, please refer to: 
# https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/pipelines
# Feel free to try out different parameters in fill-mask pipeline function.
classifier = pipeline("fill-mask", model=model, tokenizer=tokenizer, targets=targets)
cb_score = 0


def replace_template(template, token):
  # token is either "[MASK]" or "attribute"
  template = template.replace("[TTT]", "[MASK]")
  return template.replace("[AAA]", token) if "[AAA]" in template else template.replace("[AAAs]", pluralize(token))


variances = [[0] * len(attributes)] * len(templates)

# iter templates
for i, template in enumerate(templates):
  prior_sentence = replace_template(template, "[MASK]")
  prior_prediction = classifier(prior_sentence, top_k=len(targets))[0]
  sorted_prior = sorted(prior_prediction, key=lambda x: x['token_str'])

  target_sentences = [replace_template(template, attribute) for attribute in attributes]

  # Under a specific template..
  # Sum of normalized probabilities on different attributes, for each country
  template_norm_probs = {}

  # iter attributes
  for j, target_sentence in enumerate(target_sentences):
    print(f"prior_sentence: {prior_sentence}\ntarget_sentence: {target_sentence}")
    target_prediction = classifier(target_sentence, top_k=len(targets))
    sorted_target = sorted(target_prediction, key=lambda x: x['token_str'])

    norm_probs = []
    # iter countries (ethnicity)
    for prior, target in zip(sorted_prior, sorted_target):
      token = target['token']
      country = target['token_str']
      sequence = '[MASK]'.join(target['sequence'].split(country))
      norm_prob = target['score'] / prior['score']
      norm_probs.append((norm_prob, token, country, sequence))
    
    for norm_prob, _, country, _ in norm_probs:
      template_norm_probs[country] = template_norm_probs.get(country, 0) + norm_prob

    norm_probs.sort(key=lambda x: x[0])
    norm_probs.reverse()
    # pprint(norm_probs[:])
    
    logPs = [math.log(norm_prob) for norm_prob, token, country, sequence in norm_probs]
    variance = sum([e ** 2 for e in logPs])/len(logPs) - (sum(logPs) / len(logPs)) ** 2
    cb_score += variance
    variances[i][j] = (variance, target_sentence)
    # print(f"variance: {variance}")

  print("\n[Average score of different attributes for each contry]")
  template_norm_probs = list(template_norm_probs.items())
  template_norm_probs.sort(key=lambda x: x[1])
  template_norm_probs.reverse()
  pprint(template_norm_probs)

cb_score = cb_score / len(templates) / len(attributes)
print(f"cb_score: {cb_score}")
print(f"variances: {variances}")

print("\n[Average score of different templates for each attribute]")
for attribute_idx, attribute in enumerate(attributes):
  print(f"Attribute: {attribute}")
  variances_avg = 0
  for template_list in variances:
    variance, target_sentence = template_list[attribute_idx]
    variances_avg += variance
  variances_avg = variances_avg / len(templates)
  print(f"Average variances: {variances_avg}")
