# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import math
import time
import itertools
from tqdm import tqdm
import torch
from torch import nn
from laserembeddings import Laser
from sentence_transformers import SentenceTransformer, util
from embed_bible_texts import *
from bitext_retrieval import *
from gromov_hausdorff import *
from svg import *
from preprocess_embeddings import *
from econd_hm import *
from typological_distances import *
from compute_overlap import *

# Code begins here
if __name__=='__main__':
    
  '''
  Checks if GPU is available (recommended)
  '''
  if torch.cuda.is_available():
      print('Using GPU (recommended): {}'.format(torch.cuda.get_device_name(0)))
      device = torch.device('cuda')
  else:
      print('Using CPU (not recommended)')
      device = torch.device('cpu')

  '''
  Loading LaBSE model and moving it to GPU
  '''
  labse_model = SentenceTransformer('LaBSE')
  if torch.cuda.is_available():
    labse_model.cuda()

  '''
  Loading LASER model (automatically uses GPU if available)
  '''
  laser_model = Laser()
  if torch.cuda.is_available():
      print('Using GPU for LASER embeddings')
  else:
      print('Using CPU for LASER embeddings')

  parser = argparse.ArgumentParser(description='Data generation script')
  parser.add_argument('--path_to_matthew', default='/Data/Bible bitexts/book_of_matthew',
    help='Directory where Book of Matthew texts are stored')
  parser.add_argument('--path_to_john', default='/Data/Bible bitexts/book_of_john',
    help='Directory where Book of John texts are stored')
  parser.add_argument('--download_dir', help='Directory to which DataFrame containing features is downloaded')
  parser.add_argument('--path_to_lang_df', default='/Data/Bible experimental vars/bible_all_features_LANGUAGE')
  args = parser.parse_args()

  '''
  Reading in Bible texts
  '''
  print('Reading Bible texts')
  print('Book of Matthew . . .')
  matthew_texts = readTexts(args.path_to_matthew)
  print('. . . and Book of John')
  john_texts = readTexts(args.path_to_john)

  '''
  Generating a list of languages used (utility variable)
  '''
  langs = getLangs(args.path_to_matthew)

  '''
  ISO 639-1 dictionary (utility for LASER embedding, among other tasks)
  '''

  iso_codes ={       
        "aa":    "Afar",
        "ab":    "Abkhazian",
        "af":    "Afrikaans",
        "am":    "Amharic",
        "ar":    "Arabic",
        "ar-ae": "Arabic (U.A.E.)",
        "ar-bh": "Arabic (Bahrain)",
        "ar-dz": "Arabic (Algeria)",
        "ar-eg": "Arabic (Egypt)",
        "ar-iq": "Arabic (Iraq)",
        "ar-jo": "Arabic (Jordan)",
        "ar-kw": "Arabic (Kuwait)",
        "ar-lb": "Arabic (Lebanon)",
        "ar-ly": "Arabic (Libya)",
        "ar-ma": "Arabic (Morocco)",
        "ar-om": "Arabic (Oman)",
        "ar-qa": "Arabic (Qatar)",
        "ar-sa": "Arabic (Saudi Arabia)",
        "ar-sy": "Arabic (Syria)",
        "ar-tn": "Arabic (Tunisia)",
        "ar-ye": "Arabic (Yemen)",
        "as":    "Assamese",
        "ay":    "Aymara",
        "az":    "Azeri",
        "ba":    "Bashkir",
        "be":    "Belarusian",
        "bg":    "Bulgarian",
        "bh":    "Bihari",
        "bi":    "Bislama",
        "bn":    "Bengali",
        "bo":    "Tibetan",
        "br":    "Breton",
        "ca":    "Catalan",
        "co":    "Corsican",
        "cs":    "Czech",
        "cy":    "Welsh",
        "da":    "Danish",
        "de":    "German",
        "de-at": "German (Austria)",
        "de-ch": "German (Switzerland)",
        "de-li": "German (Liechtenstein)",
        "de-lu": "German (Luxembourg)",
        "div":   "Divehi",
        "dz":    "Bhutani",
        "el":    "Greek",
        "en":    "English",
        "en-au": "English (Australia)",
        "en-bz": "English (Belize)",
        "en-ca": "English (Canada)",
        "en-gb": "English (United Kingdom)",
        "en-ie": "English (Ireland)",
        "en-jm": "English (Jamaica)",
        "en-nz": "English (New Zealand)",
        "en-ph": "English (Philippines)",
        "en-tt": "English (Trinidad)",
        "en-us": "English (United States)",
        "en-za": "English (South Africa)",
        "en-zw": "English (Zimbabwe)",
        "eo":    "Esperanto",
        "es":    "Spanish",
        "es-ar": "Spanish (Argentina)",
        "es-bo": "Spanish (Bolivia)",
        "es-cl": "Spanish (Chile)",
        "es-co": "Spanish (Colombia)",
        "es-cr": "Spanish (Costa Rica)",
        "es-do": "Spanish (Dominican Republic)",
        "es-ec": "Spanish (Ecuador)",
        "es-es": "Spanish (España)",
        "es-gt": "Spanish (Guatemala)",
        "es-hn": "Spanish (Honduras)",
        "es-mx": "Spanish (Mexico)",
        "es-ni": "Spanish (Nicaragua)",
        "es-pa": "Spanish (Panama)",
        "es-pe": "Spanish (Peru)",
        "es-pr": "Spanish (Puerto Rico)",
        "es-py": "Spanish (Paraguay)",
        "es-sv": "Spanish (El Salvador)",
        "es-us": "Spanish (United States)",
        "es-uy": "Spanish (Uruguay)",
        "es-ve": "Spanish (Venezuela)",
        "et":    "Estonian",
        "eu":    "Basque",
        "fa":    "Farsi",
        "fi":    "Finnish",
        "fj":    "Fiji",
        "fo":    "Faeroese",
        "fr":    "French",
        "fr-be": "French (Belgium)",
        "fr-ca": "French (Canada)",
        "fr-ch": "French (Switzerland)",
        "fr-lu": "French (Luxembourg)",
        "fr-mc": "French (Monaco)",
        "fy":    "Frisian",
        "ga":    "Irish",
        "gd":    "Gaelic",
        "gl":    "Galician",
        "gn":    "Guarani",
        "gu":    "Gujarati",
        "ha":    "Hausa",
        "he":    "Hebrew",
        "hi":    "Hindi",
        "hr":    "Croatian",
        "hu":    "Hungarian",
        "hy":    "Armenian",
        "ia":    "Interlingua",
        "id":    "Indonesian",
        "ie":    "Interlingue",
        "ik":    "Inupiak",
        "in":    "Indonesian",
        "is":    "Icelandic",
        "it":    "Italian",
        "it-ch": "Italian (Switzerland)",
        "iw":    "Hebrew",
        "ja":    "Japanese",
        "ji":    "Yiddish",
        "jw":    "Javanese",
        "ka":    "Georgian",
        "kk":    "Kazakh",
        "kl":    "Greenlandic",
        "km":    "Cambodian",
        "kn":    "Kannada",
        "ko":    "Korean",
        "kok":   "Konkani",
        "ks":    "Kashmiri",
        "ku":    "Kurdish",
        "ky":    "Kirghiz",
        "kz":    "Kyrgyz",
        "la":    "Latin",
        "ln":    "Lingala",
        "lo":    "Laothian",
        "ls":    "Slovenian",
        "lt":    "Lithuanian",
        "lv":    "Latvian",
        "mg":    "Malagasy",
        "mi":    "Maori",
        "mk":    "FYRO Macedonian",
        "ml":    "Malayalam",
        "mn":    "Mongolian",
        "mo":    "Moldavian",
        "mr":    "Marathi",
        "ms":    "Malay",
        "mt":    "Maltese",
        "my":    "Burmese",
        "na":    "Nauru",
        "nb-no": "Norwegian (Bokmal)",
        "ne":    "Nepali (India)",
        "nl":    "Dutch",
        "nl-be": "Dutch (Belgium)",
        "nn-no": "Norwegian",
        "no":    "Norwegian (Bokmal)",
        "oc":    "Occitan",
        "om":    "(Afan)/Oromoor/Oriya",
        "or":    "Oriya",
        "pa":    "Punjabi",
        "pl":    "Polish",
        "ps":    "Pashto/Pushto",
        "pt":    "Portuguese",
        "pt-br": "Portuguese (Brazil)",
        "qu":    "Quechua",
        "rm":    "Rhaeto-Romanic",
        "rn":    "Kirundi",
        "ro":    "Romanian",
        "ro-md": "Romanian (Moldova)",
        "ru":    "Russian",
        "ru-md": "Russian (Moldova)",
        "rw":    "Kinyarwanda",
        "sa":    "Sanskrit",
        "sb":    "Sorbian",
        "sd":    "Sindhi",
        "sg":    "Sangro",
        "sh":    "Serbo-Croatian",
        "si":    "Singhalese",
        "sk":    "Slovak",
        "sl":    "Slovenian",
        "sm":    "Samoan",
        "sn":    "Shona",
        "so":    "Somali",
        "sq":    "Albanian",
        "sr":    "Serbian",
        "ss":    "Siswati",
        "st":    "Sesotho",
        "su":    "Sundanese",
        "sv":    "Swedish",
        "sv-fi": "Swedish (Finland)",
        "sw":    "Swahili",
        "sx":    "Sutu",
        "syr":    "Syriac",
        "ta":    "Tamil",
        "te":    "Telugu",
        "tg":    "Tajik",
        "th":    "Thai",
        "ti":    "Tigrinya",
        "tk":    "Turkmen",
        "tl":    "Tagalog",
        "tn":    "Tswana",
        "to":    "Tonga",
        "tr":    "Turkish",
        "ts":    "Tsonga",
        "tt":    "Tatar",
        "tw":    "Twi",
        "uk":    "Ukrainian",
        "ur":    "Urdu",
        "us":    "English",
        "uz":    "Uzbek",
        "vi":    "Vietnamese",
        "vo":    "Volapuk",
        "wo":    "Wolof",
        "xh":    "Xhosa",
        "yi":    "Yiddish",
        "yo":    "Yoruba",
        "zh":    "Chinese",
        "zh-cn": "Chinese (China)",
        "zh-hk": "Chinese (Hong Kong SAR)",
        "zh-mo": "Chinese (Macau SAR)",
        "zh-sg": "Chinese (Singapore)",
        "zh-tw": "Chinese (Taiwan)",
        "zu":    "Zulu"}

  # Reverse the key-value relation of the dictionary
  iso_codes = {iso_codes[k]:k for k in iso_codes}  

  iso_codes = [iso_codes[lang] if lang in iso_codes else None for lang in langs]    

  '''
  Generating LaBSE embeddings
  '''
  print('Generating LaBSE embeddings')
  matthew_embs_labse = embedLaBSE(labse_model, matthew_texts)
  john_embs_labse = embedLaBSE(labse_model, john_texts)

  '''
  Generating LASER embeddings
  '''
  print('Generating LASER embeddings')
  matthew_embs_laser = embedLASER(laser_model, matthew_texts, iso_codes)
  john_embs_laser = embedLASER(laser_model, john_texts, iso_codes)

  '''
  Performing bitext retrieval
  '''
  # Ground-truth sentence alignments
  gold_pairs_matthew = [(i+1,i+1) for i in range(len(matthew_texts[0]))]
  gold_pairs_john = [(i+1,i+1) for i in range(len(john_texts[0]))]

  print('Performing bitext retrieval . . .')
  start = time.time()
  print('Bitext retrieval: Book of Matthew (LaBSE)')
  matthew_labse_f1 = [(computeF1(mineSentencePairs(x,y)[0], gold_pairs_matthew))[0] 
                      for x,y in tqdm(itertools.combinations(matthew_embs_labse, 2))]
  print('Bitext retrieval: Book of John (LaBSE)')
  john_labse_f1 = [(computeF1(mineSentencePairs(x,y)[0], gold_pairs_john))[0] 
                  for x,y in tqdm(itertools.combinations(john_embs_labse, 2))]
  print('Bitext retrieval: Book of Matthew (LASER)')
  matthew_laser_f1 = [(computeF1(mineSentencePairs(x,y)[0], gold_pairs_matthew))[0] 
                      for x,y in tqdm(itertools.combinations(matthew_embs_laser, 2))]
  print('Bitext retrieval: Book of John (LASER)')
  john_laser_f1 = [(computeF1(mineSentencePairs(x,y)[0], gold_pairs_john))[0] 
                  for x,y in tqdm(itertools.combinations(john_embs_laser, 2))]
  print('Bitext retrieval complete')
  end = time.time()
  print('Time taken: {:.2f} seconds'.format(end-start))

  '''
  Computing average margin scores
  '''
  print('Computing of average margin scores . . .')
  start = time.time()
  print('Average margin scores: Book of Matthew (LaBSE)')
  matthew_labse_avg_margin = [mineSentencePairs(x,y,average=True) for x,y in 
                          tqdm(itertools.combinations(matthew_embs_labse, 2))]
  print('Average margin scores: Book of John (LaBSE)')
  john_labse_avg_margin = [mineSentencePairs(x,y,average=True) for x,y in 
                          tqdm(itertools.combinations(john_embs_labse, 2))]
  print('Average margin scores: Book of Matthew (LASER)')
  matthew_laser_avg_margin = [mineSentencePairs(x,y,average=True) for x,y in 
                          tqdm(itertools.combinations(matthew_embs_laser, 2))]
  print('Average margin scores: Book of John (LASER)')
  john_laser_avg_margin = [mineSentencePairs(x,y,average=True) for x,y in 
                          tqdm(itertools.combinations(john_embs_laser, 2))]
  print('Average margin scores computation complete')
  end = time.time()
  print('Time taken: {:.2f} seconds'.format(end-start))


  '''
  Preprocessing embeddings for isomorphism computations
  '''
  proc_matthew_embs_labse = [preprocess_embeddings(embs) for embs in tqdm(matthew_embs_labse)]
  proc_john_embs_labse = [preprocess_embeddings(embs) for embs in tqdm(john_embs_labse)]
  proc_matthew_embs_laser = [preprocess_embeddings(embs) for embs in tqdm(matthew_embs_laser)]
  proc_john_embs_laser = [preprocess_embeddings(embs) for embs in tqdm(john_embs_laser)]

  '''
  Computing Gromov-Hausdorff distances
  '''
  print('Computing Gromov-Hausdorff distances . . .')
  start = time.time()
  print('Computing distance matrices')
  matthew_dist_mats_labse = [distance_matrix(embs) for embs in tqdm(proc_matthew_embs_labse)]
  john_dist_mats_labse = [distance_matrix(embs) for embs in tqdm(proc_john_embs_labse)]
  matthew_dist_mats_laser = [distance_matrix(embs) for embs in tqdm(proc_matthew_embs_laser)]
  john_dist_mats_laser = [distance_matrix(embs) for embs in tqdm(proc_john_embs_laser)]
  print('Retrieving distances')
  matthew_gh_dists_labse = [compute_distance(l1, l2) for l1, l2 in tqdm(itertools.combinations(matthew_dist_mats_labse, 2))]
  john_gh_dists_labse = [compute_distance(l1, l2) for l1, l2 in tqdm(itertools.combinations(john_dist_mats_labse, 2))]
  matthew_gh_dists_laser = [compute_distance(l1, l2) for l1, l2 in tqdm(itertools.combinations(matthew_dist_mats_laser, 2))]
  john_gh_dists_laser = [compute_distance(l1, l2) for l1, l2 in tqdm(itertools.combinations(john_dist_mats_laser, 2))]
  print('Gromov-Hausdorff computation complete')
  end = time.time()
  print('Time taken: {:.2f} seconds'.format(end-start))

  '''
  Computing singular value gaps
  '''
  print('Computing singular value gaps . . .')
  start = time.time()
  matthew_svg_labse = computeSVG(proc_matthew_embs_labse)
  john_svg_labse = computeSVG(proc_john_embs_labse)
  matthew_svg_laser = computeSVG(proc_matthew_embs_laser)
  john_svg_laser = computeSVG(proc_john_embs_laser)
  print('Singular value gap computation complete')
  end = time.time()
  print('Time taken: {:.2f} seconds'.format(end-start))

  '''
  Computing ECOND-HMs
  '''
  print('Computing ECOND-HMs . . .')
  start = time.time()
  matthew_econdhm_labse = computeECOND_HM(proc_matthew_embs_labse)
  john_econdhm_labse = computeECOND_HM(proc_john_embs_labse)
  matthew_econdhm_laser = computeECOND_HM(proc_matthew_embs_laser)
  john_econdhm_laser = computeECOND_HM(proc_john_embs_laser)
  print('ECOND-HM computation complete')
  end = time.time()
  print('Time taken: {:.2f} seconds'.format(end-start))

  '''
  Computing character-level overlaps
  '''
  print('Computing character-level overlaps . . .')
  matthew_chars = [getChars(text) for text in matthew_texts]
  char_overlaps = [getCharOverlap(l1,l2) for l1,l2 in tqdm(itertools.combinations(matthew_chars, 2))]

  '''
  Computing token-level overlaps
  '''
  print('Computing token-level overlaps . . .')
  # Tokenize sentences in Book of John
  john_tokenized = [_tokenize(text) for text in john_texts]
  # Pull out subword token IDs
  john_tok_ids = [getTokenIDs(tok_text) for tok_text in tqdm(john_tokenized)]
  # Get pairwise token-level overlap for all language pairs
  token_overlaps = [getTokenOverlap(l1, l2) for 
                    l1,l2 in tqdm(itertools.combinations(john_tok_ids, 2))]

  '''
  Computing typological distances 
  '''
  print('Computing typological distances . . .')
  start = time.time()
  lang_df = pd.read_csv(args.path_to_lang_df)
  bible_iso_codes = lang_df['ISO 639-3']
  bible_iso_codes = list(bible_iso_codes.replace('arc', 'heb'))
  ldf_langs = list(lang_df.Language)
  for i in range(len(ldf_langs)):
    if ldf_langs[i]=='Haitian Creole': ldf_langs[i]='Creole'
  for i in range(len(ldf_langs)):
    if ldf_langs[i]!=sorted(langs)[i]:
        ldf_langs[i] = sorted(langs)[i]
  bible_iso_codes.remove('hat')
  bible_iso_codes.insert(22, 'hat')
  bible_iso_codes = {l:c for l,c in zip(ldf_langs, bible_iso_codes)}
  bible_iso_codes = [bible_iso_codes[l] for l in langs if l in bible_iso_codes]

  geo_dists = getTypologicalDistances(bible_iso_codes, dist_type='geo')
  syn_dists = getTypologicalDistances(bible_iso_codes, dist_type='syntax_knn')
  phon_dists = getTypologicalDistances(bible_iso_codes, dist_type='phonology_knn')
  inv_dists = getTypologicalDistances(bible_iso_codes, dist_type='inventory_knn')
  print('Typological distance computations complete')
  end = time.time()
  print('Time taken: {:.2f} seconds'.format(end-start))

  '''
  Writing results to a file
  '''
  print('Writing results to a CSV')
  df_dict = {'F1-score (LaBSE, Book of Matthew)': matthew_labse_f1,
             'F1-score (LaBSE, Book of John'): john_labse_f1,
             'F1-score (LASER, Book of Matthew)': matthew_laser_f1,
             'F1-score (LASER, Book of John)': john_laser_f1,
             'Avg. margin score (LaBSE, Book of Matthew)': matthew_labse_avg_margin,
             'Avg. margin score (LaBSE, Book of John)': john_labse_avg_margin,
             'Avg margin score (LASER, Book of Matthew)': matthew_laser_avg_margin,
             'Avg. margin score (LASER, Book of John)': john_laser_avg_margin,
             'Gromov-Hausdorff distance (LaBSE, Book of Matthew)': matthew_gh_dists_labse,
             'Gromov-Hausdorff distance (LaBSE, Book of John)': john_gh_dists_labse,
             'Gromov-Hausdorff distance (LASER, Book of Matthew)': matthew_gh_dists_laser,
             'Gromov-Hausdorff distance (LASER, Book of John)': john_gh_dists_laser,
             'SVG (LaBSE, Book of Matthew)': matthew_svg_labse,
             'SVG (LaBSE, Book of John)': john_svg_labse,
             'SVG (LASER, Book of Matthew)': matthew_svg_laser,
             'SVG (LASER, Book of John)': john_svg_laser,
             'ECOND-HM (LaBSE, Book of Matthew)': matthew_econdhm_labse,
             'ECOND-HM (LaBSE, Book of John)': john_econdhm_labse,
             'ECOND-HM (LASER, Book of Matthew)': matthew_econdhm_laser,
             'ECOND-HM (LASER, Book of John)': john_econdhm_laser,
             'Character-level overlap': char_overlaps,
             'Token-level overlap': token_overlaps,
             'Geographic distance': geo_dists,
             'Syntactic distance': syn_dists,
             'Phonological distance': phon_dists,
             'Inventory distance': inv_dists}
             
  pd.DataFrame(df_dict).to_csv(args.download_dir)
