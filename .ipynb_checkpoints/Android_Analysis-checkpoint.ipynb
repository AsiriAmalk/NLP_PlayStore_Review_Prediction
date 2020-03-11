{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Android App Rating Analyer \n",
    "* Model - Google Bert\n",
    "* Inputs - Rate, Comment\n",
    "* ~100K data records"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "######################################################################\n",
    "### Auther: Asiri Amal K                                          ####\n",
    "### Model : Google's BERT                                         ####\n",
    "######################################################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "import torch\n",
    "import pickle\n",
    "from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)\n",
    "from torch.nn import CrossEntropyLoss, MSELoss\n",
    "\n",
    "from tqdm import tqdm_notebook, trange\n",
    "import os\n",
    "from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification\n",
    "from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule\n",
    "\n",
    "from multiprocessing import Pool, cpu_count\n",
    "from run_classifier import *\n",
    "import convert_examples_to_features\n",
    "\n",
    "# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows\n",
    "import logging\n",
    "logging.basicConfig(level=logging.INFO)\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "rates = pd.read_csv(\"android_reviews_100k.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# The input data dir. Should contain the .tsv files (or other data files) for the task.\n",
    "DATA_DIR = \"data/\"\n",
    "\n",
    "# Bert pre-trained model selected in the list: bert-base-uncased, \n",
    "# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,\n",
    "# bert-base-multilingual-cased, bert-base-chinese.\n",
    "BERT_MODEL = 'yelp.tar.gz'\n",
    "\n",
    "# The name of the task to train.I'm going to name this 'yelp'.\n",
    "TASK_NAME = 'yelp'\n",
    "\n",
    "# The output directory where the fine-tuned model and checkpoints will be written.\n",
    "OUTPUT_DIR = f'outputs/{TASK_NAME}/'\n",
    "\n",
    "# The directory where the evaluation reports will be written to.\n",
    "REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_reports/'\n",
    "\n",
    "# This is where BERT will look for pre-trained models to load parameters from.\n",
    "CACHE_DIR = 'cache/'\n",
    "\n",
    "# The maximum total input sequence length after WordPiece tokenization.\n",
    "# Sequences longer than this will be truncated, and sequences shorter than this will be padded.\n",
    "MAX_SEQ_LENGTH = 128\n",
    "\n",
    "TRAIN_BATCH_SIZE = 24\n",
    "EVAL_BATCH_SIZE = 8\n",
    "LEARNING_RATE = 2e-5\n",
    "NUM_TRAIN_EPOCHS = 1\n",
    "RANDOM_SEED = 42\n",
    "GRADIENT_ACCUMULATION_STEPS = 1\n",
    "WARMUP_PROPORTION = 0.1\n",
    "OUTPUT_MODE = 'classification'\n",
    "\n",
    "CONFIG_NAME = \"config.json\"\n",
    "WEIGHTS_NAME = \"pytorch_model.bin\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Check the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>rating</th>\n",
       "      <th>review</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>anyone know how to get FM tuner on this launch...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>Developers of this app need to work hard to fi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>This app works great on my joying Android base...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>Shouldn't of paid for the full version and sho...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>i love the fact that it turns you phone into a...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  rating                                             review\n",
       "0           0       4  anyone know how to get FM tuner on this launch...\n",
       "1           1       2  Developers of this app need to work hard to fi...\n",
       "2           2       4  This app works great on my joying Android base...\n",
       "3           3       1  Shouldn't of paid for the full version and sho...\n",
       "4           4       5  i love the fact that it turns you phone into a..."
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rates.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(99995, 3)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# We have exact 99995 records\n",
    "rates.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Columns are\n",
    "1. Unnamed: 0\n",
    "2. rating\n",
    "3. review"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Unnamed: 0', 'rating', 'review'], dtype='object')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rates.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### First Column is unnecessary it only includes the line number so it should be removed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# rates = rates.drop('Unnamed: 0', axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>rating</th>\n",
       "      <th>review</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>anyone know how to get FM tuner on this launch...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>Developers of this app need to work hard to fi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>This app works great on my joying Android base...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>Shouldn't of paid for the full version and sho...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>i love the fact that it turns you phone into a...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  rating                                             review\n",
       "0           0       4  anyone know how to get FM tuner on this launch...\n",
       "1           1       2  Developers of this app need to work hard to fi...\n",
       "2           2       4  This app works great on my joying Android base...\n",
       "3           3       1  Shouldn't of paid for the full version and sho...\n",
       "4           4       5  i love the fact that it turns you phone into a..."
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rates.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Checking the missing values\n",
    "\n",
    "* No missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Unnamed: 0    0\n",
       "rating        0\n",
       "review        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rates.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Checking the outliers\n",
    "\n",
    "###### Ratings must between 1-5, If any number will be an outlier\n",
    "* No outliers is detected in the dataset \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5    40841\n",
       "1    25215\n",
       "4    14624\n",
       "3    11267\n",
       "2     8048\n",
       "Name: rating, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rates[\"rating\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "id_df = pd.DataFrame(rates['Unnamed: 0'].values, columns=[\"id\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id\n",
       "0   0\n",
       "1   1\n",
       "2   2\n",
       "3   3\n",
       "4   4"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "id_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "rate_df = rates[\"review\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Encode the label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>review</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>anyone know how to get FM tuner on this launch...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Developers of this app need to work hard to fi...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>This app works great on my joying Android base...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>Shouldn't of paid for the full version and sho...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>i love the fact that it turns you phone into a...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id                                             review  1  2  3  4  5\n",
       "0   0  anyone know how to get FM tuner on this launch...  0  0  0  1  0\n",
       "1   1  Developers of this app need to work hard to fi...  0  1  0  0  0\n",
       "2   2  This app works great on my joying Android base...  0  0  0  1  0\n",
       "3   3  Shouldn't of paid for the full version and sho...  1  0  0  0  0\n",
       "4   4  i love the fact that it turns you phone into a...  0  0  0  0  1"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelBinarizer\n",
    "encoder = LabelBinarizer()\n",
    "rates_cat_1hot = encoder.fit_transform(rates[\"rating\"])\n",
    "rating_df = pd.DataFrame(rates_cat_1hot, columns=encoder.classes_)\n",
    "df = pd.concat([id_df, rate_df, rating_df], axis=1)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['id', 'review', 1, 2, 3, 4, 5], dtype='object')"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['labels'] = list(zip(df[1].tolist(), df[2].tolist(), df[3].tolist(), df[4].tolist(),  df[5].tolist()))\n",
    "df['text'] = df['review'].apply(lambda x: x.replace('\\n', ' '))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Train, Test Split\n",
    "\n",
    "* Test set will be 0.2 from the original dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "\n",
    "train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(79996, 9)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(19999, 9)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Training set size: 89995\n",
    "# Test set size    : 10000\n",
    "print(train_df.shape)\n",
    "eval_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>review</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>labels</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>92722</th>\n",
       "      <td>19</td>\n",
       "      <td>It's more fun and great  if you can add cut an...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>(0, 0, 0, 0, 1)</td>\n",
       "      <td>It's more fun and great  if you can add cut an...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25159</th>\n",
       "      <td>5</td>\n",
       "      <td>The great game I have played</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>(0, 0, 0, 0, 1)</td>\n",
       "      <td>The great game I have played</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>38240</th>\n",
       "      <td>12</td>\n",
       "      <td>Thx For making this app</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>(0, 0, 0, 0, 1)</td>\n",
       "      <td>Thx For making this app</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3806</th>\n",
       "      <td>0</td>\n",
       "      <td>Have to start at begining of 1st level when fa...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>(0, 0, 1, 0, 0)</td>\n",
       "      <td>Have to start at begining of 1st level when fa...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>67881</th>\n",
       "      <td>30</td>\n",
       "      <td>This app is usefull.Its a successful  app and ...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>(0, 0, 0, 0, 1)</td>\n",
       "      <td>This app is usefull.Its a successful  app and ...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       id                                             review  1  2  3  4  5  \\\n",
       "92722  19  It's more fun and great  if you can add cut an...  0  0  0  0  1   \n",
       "25159   5                       The great game I have played  0  0  0  0  1   \n",
       "38240  12                            Thx For making this app  0  0  0  0  1   \n",
       "3806    0  Have to start at begining of 1st level when fa...  0  0  1  0  0   \n",
       "67881  30  This app is usefull.Its a successful  app and ...  0  0  0  0  1   \n",
       "\n",
       "                labels                                               text  \n",
       "92722  (0, 0, 0, 0, 1)  It's more fun and great  if you can add cut an...  \n",
       "25159  (0, 0, 0, 0, 1)                       The great game I have played  \n",
       "38240  (0, 0, 0, 0, 1)                            Thx For making this app  \n",
       "3806   (0, 0, 1, 0, 0)  Have to start at begining of 1st level when fa...  \n",
       "67881  (0, 0, 0, 0, 1)  This app is usefull.Its a successful  app and ...  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>review</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>labels</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>33967</th>\n",
       "      <td>30</td>\n",
       "      <td>This theme is very nyc.</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>(0, 0, 1, 0, 0)</td>\n",
       "      <td>This theme is very nyc.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31049</th>\n",
       "      <td>5</td>\n",
       "      <td>Blink as an app is stuck on the authentication...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>(0, 0, 0, 1, 0)</td>\n",
       "      <td>Blink as an app is stuck on the authentication...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>53746</th>\n",
       "      <td>32</td>\n",
       "      <td>There's a possible loophole in expertise mayb ...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>(0, 0, 0, 0, 1)</td>\n",
       "      <td>There's a possible loophole in expertise mayb ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9732</th>\n",
       "      <td>5</td>\n",
       "      <td>I have Internet and every thing works like a c...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>(1, 0, 0, 0, 0)</td>\n",
       "      <td>I have Internet and every thing works like a c...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7129</th>\n",
       "      <td>39</td>\n",
       "      <td>Nice! But updates intermittently. Love the loo...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>(0, 0, 0, 1, 0)</td>\n",
       "      <td>Nice! But updates intermittently. Love the loo...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       id                                             review  1  2  3  4  5  \\\n",
       "33967  30                            This theme is very nyc.  0  0  1  0  0   \n",
       "31049   5  Blink as an app is stuck on the authentication...  0  0  0  1  0   \n",
       "53746  32  There's a possible loophole in expertise mayb ...  0  0  0  0  1   \n",
       "9732    5  I have Internet and every thing works like a c...  1  0  0  0  0   \n",
       "7129   39  Nice! But updates intermittently. Love the loo...  0  0  0  1  0   \n",
       "\n",
       "                labels                                               text  \n",
       "33967  (0, 0, 1, 0, 0)                            This theme is very nyc.  \n",
       "31049  (0, 0, 0, 1, 0)  Blink as an app is stuck on the authentication...  \n",
       "53746  (0, 0, 0, 0, 1)  There's a possible loophole in expertise mayb ...  \n",
       "9732   (1, 0, 0, 0, 0)  I have Internet and every thing works like a c...  \n",
       "7129   (0, 0, 0, 1, 0)  Nice! But updates intermittently. Love the loo...  "
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "eval_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Here we are using multilable classification model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:transformers.file_utils:PyTorch version 1.3.1 available.\n",
      "INFO:transformers.configuration_utils:loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json from cache at C:\\Users\\PC\\.cache\\torch\\transformers\\e1a2a406b5a05063c31f4dfdee7608986ba7c6393f7f79db5e69dcd197208534.9dad9043216064080cf9dd3711c53c0f11fe2b09313eaa66931057b4bdcaf068\n",
      "INFO:transformers.configuration_utils:Model config {\n",
      "  \"attention_probs_dropout_prob\": 0.1,\n",
      "  \"finetuning_task\": null,\n",
      "  \"hidden_act\": \"gelu\",\n",
      "  \"hidden_dropout_prob\": 0.1,\n",
      "  \"hidden_size\": 768,\n",
      "  \"initializer_range\": 0.02,\n",
      "  \"intermediate_size\": 3072,\n",
      "  \"is_decoder\": false,\n",
      "  \"layer_norm_eps\": 1e-05,\n",
      "  \"max_position_embeddings\": 514,\n",
      "  \"num_attention_heads\": 12,\n",
      "  \"num_hidden_layers\": 12,\n",
      "  \"num_labels\": 5,\n",
      "  \"output_attentions\": false,\n",
      "  \"output_hidden_states\": false,\n",
      "  \"output_past\": true,\n",
      "  \"pruned_heads\": {},\n",
      "  \"torchscript\": false,\n",
      "  \"type_vocab_size\": 1,\n",
      "  \"use_bfloat16\": false,\n",
      "  \"vocab_size\": 50265\n",
      "}\n",
      "\n",
      "INFO:transformers.tokenization_utils:loading file https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json from cache at C:\\Users\\PC\\.cache\\torch\\transformers\\d0c5776499adc1ded22493fae699da0971c1ee4c2587111707a4d177d20257a2.ef00af9e673c7160b4d41cfda1f48c5f4cba57d5142754525572a846a1ab1b9b\n",
      "INFO:transformers.tokenization_utils:loading file https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt from cache at C:\\Users\\PC\\.cache\\torch\\transformers\\b35e7cd126cd4229a746b5d5c29a749e8e84438b14bcdb575950584fe33207e8.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda\n",
      "INFO:transformers.modeling_utils:loading weights file https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin from cache at C:\\Users\\PC\\.cache\\torch\\transformers\\228756ed15b6d200d7cb45aaef08c087e2706f54cb912863d2efe07c89584eb7.49b88ba7ec2c26a7558dda98ca3884c3b80fa31cf43a1b1f23aef3ff81ba344e\n",
      "INFO:transformers.modeling_utils:Weights of RobertaForMultiLabelSequenceClassification not initialized from pretrained model: ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']\n",
      "INFO:transformers.modeling_utils:Weights from pretrained model not used in RobertaForMultiLabelSequenceClassification: ['lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.weight']\n"
     ]
    }
   ],
   "source": [
    "from simpletransformers.classification import MultiLabelClassificationModel\n",
    "\n",
    "\n",
    "model = MultiLabelClassificationModel('roberta', 'roberta-base', num_labels=5, args={'train_batch_size':2, 'gradient_accumulation_steps':16, 'learning_rate': 3e-5, 'num_train_epochs': 3, 'max_seq_length': 512})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Converting to features started.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cef1002606464270b5388cec3b1d75cb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=79996), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "77d074bc60804f1a84f6a8c378cf3246",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, description='Epoch', max=3, style=ProgressStyle(description_width='initial…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0e1bbf48782f46adb8ebff86e5d75b8c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, description='Current iteration', max=39998, style=ProgressStyle(descriptio…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running loss: 0.353541"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:transformers.configuration_utils:Configuration saved in outputs/checkpoint-2000\\config.json\n",
      "INFO:transformers.modeling_utils:Model weights saved in outputs/checkpoint-2000\\pytorch_model.bin\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running loss: 0.393916"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d6fb35a9884445caa0a569c88e73e118",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, description='Current iteration', max=39998, style=ProgressStyle(descriptio…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running loss: 0.199047"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:transformers.configuration_utils:Configuration saved in outputs/checkpoint-4000\\config.json\n",
      "INFO:transformers.modeling_utils:Model weights saved in outputs/checkpoint-4000\\pytorch_model.bin\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running loss: 0.275296"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "047d668cd6d24f0dafafcefe84c1c53e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, description='Current iteration', max=39998, style=ProgressStyle(descriptio…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running loss: 0.180899"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:transformers.configuration_utils:Configuration saved in outputs/checkpoint-6000\\config.json\n",
      "INFO:transformers.modeling_utils:Model weights saved in outputs/checkpoint-6000\\pytorch_model.bin\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running loss: 0.136548\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:transformers.configuration_utils:Configuration saved in outputs/config.json\n",
      "INFO:transformers.modeling_utils:Model weights saved in outputs/pytorch_model.bin\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training of roberta model complete. Saved to outputs/.\n"
     ]
    }
   ],
   "source": [
    "model.train_model(train_df=train_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Features loaded from cache at cache_dir/cached_dev_roberta_512_5_19999\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6e5f427ff7d244e7b989efff775c894a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=2500), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "result, model_outputs, wrong_predictions = model.eval_model(eval_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Converting to features started.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e6f803e52c2746a693daeeee467fcaab",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8f90babc1ec64eafb272cfc960793ad4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=13), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "#Lets Check and save the values for first 100 records\n",
    "test_df = rates[100:200]\n",
    "\n",
    "to_predict = test_df.review.apply(lambda x: x.replace('\\n', ' ')).tolist()\n",
    "preds, outputs = model.predict(to_predict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check the output/submission.csv file to check the values\n",
    "sub_df = pd.DataFrame(outputs, columns=[1, 2, 3, 4, 5])\n",
    "sub_df = sub_df[[ 1, 2, 3, 4, 5]]\n",
    "sub_df.to_csv('outputs/submission.csv', index=False)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### By default, only the Label ranking average precision  (LARP)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'LRAP': 0.7975282097438486, 'eval_loss': 0.3087493131391704}"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model output gives the probability of each class between 0-1 the maximum probability is the predicted class in here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.04453607, 0.0145817 , 0.0596751 , 0.12938076, 0.77890325],\n",
       "       [0.90409946, 0.07307938, 0.01934159, 0.00535511, 0.00813462],\n",
       "       [0.05863893, 0.11693754, 0.42954332, 0.24719469, 0.13259025],\n",
       "       ...,\n",
       "       [0.74486136, 0.19882403, 0.05863896, 0.00819906, 0.01139031],\n",
       "       [0.00207611, 0.00149259, 0.00843326, 0.15627551, 0.8533682 ],\n",
       "       [0.00365104, 0.00363345, 0.04206998, 0.74054515, 0.21628873]],\n",
       "      dtype=float32)"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_outputs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### We are using 1-5 as our rating labels \n",
    "### Probaliity list has 0-4 as indexing\n",
    "### We can have the location of the maximum probability index by `argmax` funciton"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get Predictions\n",
    "\n",
    "pred_ratings = []\n",
    "for i in model_outputs:\n",
    "    pred_ratings.append(i.argmax()+1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[5, 1, 3, 3, 4, 3, 5, 1, 5, 5]"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# First 10 valus\n",
    "pred_ratings[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(19999,)"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# pred_ratings\n",
    "eval_df[\"labels\"].shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Get eval set as test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "original_ratings = []\n",
    "for i in eval_df[\"labels\"]:\n",
    "    original_ratings.append(np.array(i).argmax()+1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[3, 4, 5, 1, 4, 2, 5, 1, 5, 5]"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# original_ratings\n",
    "# First 10 valus\n",
    "original_ratings[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_predict\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import precision_score, recall_score\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [],
   "source": [
    "def scores(l, p):\n",
    "    acc = accuracy_score(l, p)\n",
    "    print(\"Accuracy  :  {:.2f} %\".format(acc*100))\n",
    "    pr = precision_score(l, p, average='macro')\n",
    "    re = recall_score(l, p , average='macro')\n",
    "    f1_s = f1_score(l, p, average='macro')\n",
    "    print(\"Precision :  {:.2f} \".format(pr))\n",
    "    print(\"Recall    :  {:.2f} \".format(re))\n",
    "    print(\"F1_Score  :  {:.2f} \".format(f1_s))\n",
    "    conf_mx = confusion_matrix(l, p)\n",
    "    print(\"Confusion Matrix  :  \\n{} \".format(conf_mx))\n",
    "    plt.figure(figsize=(10, 10))\n",
    "    plt.matshow(conf_mx, fignum=1, cmap=plt.cm.binary)\n",
    "    plt.show()\n",
    "    return acc*100, pr, re, f1_s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy  :  66.24 %\n",
      "Precision :  0.54 \n",
      "Recall    :  0.51 \n",
      "F1_Score  :  0.52 \n",
      "Confusion Matrix  :  \n",
      "[[4081  204  299   64  384]\n",
      " [ 816  265  333   82  120]\n",
      " [ 498  206  825  418  334]\n",
      " [ 190   47  437 1065 1148]\n",
      " [ 320   17  211  624 7011]] \n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "(66.23831191559579, 0.5387177457255637, 0.5124698899639496, 0.5156205356094427)"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scores(original_ratings, pred_ratings)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion\n",
    "\n",
    "`Confusion matrix graph 0-4 indexs are the 1-5 class labels`\n",
    "\n",
    "Diagonal area is the TP values\n",
    "Rating 5 has very unique reviews than others so it can be identified as very often than any other rating value.\n",
    "\n",
    "We can see some gray areas in rating 4 also it means it is likely to be actual rating 4 is recognized as rating 5 because the model recognizes the rating 4 reviews as rating 5 reviews.  So, it means rating 4 has reviewed just pretty much like rating 5.\n",
    "Rating 1 has the grayest area after the 5 , So rating 1 has unique reviews also comparing with rating 2 and 3. Rating 1 column has more gray areas in row 2 and rows 3. So, reviews are from the rating 2 and rating 3 has relatively close reviews. \n",
    "\n",
    "Finally,\n",
    "\n",
    "1. Rating 5 has unique reviews.\n",
    "2. Rating 4 has very close reviews to rating 5 rather than rating 4 itself.\n",
    "3. Rating 3 and 2 have the same reviews with the comparison and also those includes rating 1 review parts heavily\n",
    "4. Rating 1 has some unique reviews comparing with 2 and 3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
