import pandas as pd
import numpy as np
from tensorflow import keras
from keras_bert import get_custom_objects
import ktrain
from ktrain import text

import os
from django.conf import settings
import json

def init_eec():
    eec_df=pd.read_csv(os.path.join(settings.BASE_DIR, 'Equity-Evaluation-Corpus.csv'), sep=',',header=0)
    print("csv read!")
    return eec_df

def init_predictor(path):
    predictor = ktrain.load_predictor(os.path.join(settings.BASE_DIR, path))
    print("model loaded!")
    return predictor

def predict_sent(predictor, sentence):
    predictor = predictor



    return predictor.predict(sentence, return_proba=True)

def get_sentiment(score):
  if score <= 0.43:
    label = "NEGATIVE"
  elif score >= 0.66:
    label = "POSITIVE"
  else:
    label = "NEUTRAL"
  return label

def emotion_df(df,emotion):
  new = df[df['Emotion']==emotion]
  return new[["Sentence","Race","Person","Emotion word","sentiment","probability"]]

def get_race_df(df_w, df_b, emotion):
  black_emt = emotion_df(df_b,emotion)
  white_emt = emotion_df(df_w,emotion)

  gend_emt = {"Black":[black_emt.probability.iloc[i] for i in range(len(black_emt))],
    "White": [white_emt.probability.iloc[i] for i in range(len(white_emt))],
    "Sentence": [[black_emt.Sentence.iloc[i],white_emt.Sentence.iloc[i]] for i in range(len(black_emt))]}
  gend_emt = pd.DataFrame(gend_emt)

  gend_emt.insert(2,column='difference',value=gend_emt.Black-gend_emt.White)

  return gend_emt

def get_gender_df(df_m,df_f,emotion):
  male_emt = emotion_df(df_m,emotion)
  female_emt = emotion_df(df_f,emotion)

  gender_emt = {"Female":[female_emt.probability.iloc[i] for i in range(len(female_emt))],
    "Male": [male_emt.probability.iloc[i] for i in range(len(male_emt))],
    "Sentence": [[female_emt.Sentence.iloc[i],male_emt.Sentence.iloc[i]] for i in range(len(male_emt))]}
  gender_emt = pd.DataFrame(gender_emt)

  gender_emt.insert(2,column='difference',value = gender_emt.Female-gender_emt.Male)

  return gender_emt

def gender_intensity(scores):
  if scores[0] > scores[1]:
    return "Female"
  else:
    return "Male"

def get_gender_intensity(df):
  df['probability intensity']=df.apply(gender_intensity,axis=1)

def race_intensity(scores):
  if scores[0] > scores[1]:
    return "Black"
  else:
    return "White"

def get_race_intensity(df):
  df['probability intensity']=df.apply(race_intensity,axis=1)

def scores_df(df, predictor):
  df["sentiment"] = df.Sentence.apply(lambda x: predictor.predict(x))

  df["probability"] = df.Sentence.apply(lambda x:predictor.predict(x,return_proba=True).max())
  return df

def gender_disparities(predictor, eec):
    emotion = ['ANGER','FEAR','JOY','SADNESS']
    has_eecs = [os.path.exists(os.path.join(settings.BASE_DIR, "gender_"+emt_i.lower()+".csv")) for emt_i in emotion]
    gender_anger = gender_joy = gender_fear = gender_sad = []
    if (all(has_eecs)):
        print("has eecs")
        gender_anger = pd.read_csv(os.path.join(settings.BASE_DIR, "gender_anger.csv"))
        gender_joy = pd.read_csv(os.path.join(settings.BASE_DIR, "gender_joy.csv"))
        gender_fear = pd.read_csv(os.path.join(settings.BASE_DIR, "gender_fear.csv"))
        gender_sad = pd.read_csv(os.path.join(settings.BASE_DIR, "gender_sadness.csv"))
    else:
        male_df = eec[eec['Gender']== 'male']
        female_df = eec[eec['Gender']== 'female']
        female_df = scores_df(female_df, predictor)
        male_df = scores_df(male_df, predictor)
        gender_anger = get_gender_df(male_df,female_df,'anger')
        gender_anger.to_csv("gender_anger.csv", index=False)
        gender_joy = get_gender_df(male_df,female_df,'joy')
        gender_joy.to_csv("gender_joy.csv", index=False)
        gender_fear = get_gender_df(male_df,female_df,'fear')
        gender_fear.to_csv("gender_fear.csv", index=False)
        gender_sad = get_gender_df(male_df,female_df,'sadness')
        gender_sad.to_csv("gender_sadness.csv", index=False)

    gender_dfs = [gender_anger,gender_fear,gender_joy,gender_sad]

    for df in gender_dfs:
        get_gender_intensity(df)
    
    data_summary = []
    for i,df in zip(emotion,gender_dfs):
        probability_counts = df['probability intensity'].value_counts()
        data_summary.append({"sentiment":i.lower(),"probability": {"female": int(probability_counts["Female"]), "male": int(probability_counts["Male"])}})

    return data_summary

def racial_disparities(predictor, eec):
    emotion = ['ANGER','FEAR','JOY','SADNESS']
    has_eecs = [os.path.exists(os.path.join(settings.BASE_DIR, "racial_"+emt_i.lower()+".csv")) for emt_i in emotion]
    racial_anger = racial_joy = racial_fear = racial_sad = []
    if (all(has_eecs)):
        print("has eecs")
        racial_anger = pd.read_csv(os.path.join(settings.BASE_DIR, "racial_anger.csv"))
        racial_joy = pd.read_csv(os.path.join(settings.BASE_DIR, "racial_joy.csv"))
        racial_fear = pd.read_csv(os.path.join(settings.BASE_DIR, "racial_fear.csv"))
        racial_sad = pd.read_csv(os.path.join(settings.BASE_DIR, "racial_sadness.csv"))
    else:
        b_df = eec[eec['Race']== 'African-American']
        w_df = eec[eec['Race']== 'European']
        b_df = scores_df(b_df, predictor)
        w_df = scores_df(w_df, predictor)
        racial_anger = get_race_df(w_df,b_df,'anger')
        racial_anger.to_csv("racial_anger.csv", index=False)
        racial_joy = get_race_df(w_df,b_df,'joy')
        racial_joy.to_csv("racial_joy.csv", index=False)
        racial_fear = get_race_df(w_df,b_df,'fear')
        racial_fear.to_csv("racial_fear.csv", index=False)
        racial_sad = get_race_df(w_df,b_df,'sadness')
        racial_sad.to_csv("racial_sadness.csv", index=False)

    racial_dfs = [racial_anger,racial_fear,racial_joy,racial_sad]

    for df in racial_dfs:
        get_race_intensity(df)
    
    data_summary = []
    for i,df in zip(emotion,racial_dfs):
        probability_counts = df['probability intensity'].value_counts()
        data_summary.append({"sentiment":i.lower(),"probability": {"black": int(probability_counts["Black"]), "white": int(probability_counts["White"])}})

    return data_summary
