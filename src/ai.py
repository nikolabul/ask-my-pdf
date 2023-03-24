from ai_bricks.api import openai
from sentence_transformers import SentenceTransformer
import stats
import os
import streamlit as st

from transformers import pipeline
import torch

DEFAULT_USER = 'cb232f0236c29dc5' # community user
DEVICE_GPU = torch.device('cuda')
DEVICE_CPU = torch.device('cpu')

# in comments => max_seq_len - embed_dim
# CHECKPOINT = "all-mpnet-base-v2" # 384-768
# CHECKPOINT = "multi-qa-distilbert-cos-v1" # 512-768
# CHECKPOINT = "all-MiniLM-L6-v2" # 256-384


with st.spinner('Enabling GPU and Importing local models'):

	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
	print(f"Torch version: {torch.__version__}") 
	print(f"CUDA device count: {torch.cuda.device_count()}")
	print(f"Dev 0: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'}")
	print(f"Dev 1: {torch.cuda.get_device_properties(1).name if torch.cuda.is_available() else 'CPU'}")
	print(f"CUDA version: {torch.version.cuda}")

	checkpoint = 'togethercomputer/GPT-NeoXT-Chat-Base-20B'
	pipe = pipeline(
		task="text-generation", 
		model=checkpoint, 
		tokenizer=checkpoint, 
		use_fast=True,
		# device=torch.device('cpu'),
		device_map="auto", # auto  balanced_low_0
		torch_dtype=torch.bfloat16, # torch.float16, torch.bfloat16, auto
		model_kwargs = {
			"temperature": 0,
			# "load_in_8bit": True, # torch_dtype should be set to torch.float16
			# "max_length": 4000,
			# "max_new_tokens": 100,
			# "offload_folder": "offload_folder/",
			# "offload_state_dict": True,
			# "max_memory": {0: "8GiB", 1: "22GiB", "cpu": "30GiB"},
		}
	)

	# clear cuda cache
	torch.cuda.empty_cache()


def use_key(key):
	openai.use_key(key)

usage_stats = stats.get_stats(user=DEFAULT_USER)
def set_user(user):
	global usage_stats
	usage_stats = stats.get_stats(user=user)
	openai.set_global('user', user)
	openai.add_callback('after', stats_callback)

def complete(text, **kw):
	
	model = kw.get('model','gpt-3.5-turbo')
	
	if model in ['gpt-3.5-turbo','text-davinci-003','text-curie-001']:
		llm = openai.model(model)
		llm.config['pre_prompt'] = 'output only in raw text' # for chat models
		resp = llm.complete(text, **kw)
		
	elif model in ['GPT-NeoXT-Chat-Base-20B']:
		response = pipe(f'''<human>: {text}\n<bot>:''', max_new_tokens=100)
		# print(response[0]['generated_text'])
		
		resp = {}
		resp['text'] = response[0]['generated_text']
		resp['usage'] = {'tokens': 0, 'characters': 0, 'requests': 0}

		# clear cuda cache
		torch.cuda.empty_cache()

	resp['model'] = model
	return resp

def embedding(text, model_embed, **kw):
	# model = kw.get('model_embed','text-embedding-ada-002')
	if model_embed in ['text-embedding-ada-002']:
		llm = openai.model(model_embed)
		resp = llm.embed(text, **kw)
	elif model_embed in ['all-mpnet-base-v2', 'multi-qa-distilbert-cos-v1', 'all-MiniLM-L6-v2']:
		model = SentenceTransformer(model_embed, device=DEVICE_CPU)
		# list of embeddings (texts count x embedding dim)
		embeddings = model.encode(
			sentences = text,
			batch_size = 100,
			show_progress_bar = False,
			output_value = 'sentence_embedding', # Set to None, to get all output values ('sentence_embedding', 'token_embeddings')
			device = DEVICE_CPU,
		)
		resp = {}
		resp['model_embed'] = model_embed
		resp['vector'] = embeddings.tolist()
	else:
		return

	resp['model_embed'] = model_embed
	return resp


def embeddings(texts, model_embed, **kw):
	# model = kw.get('model_embed','text-embedding-ada-002')
	if model_embed in ['text-embedding-ada-002']:
		llm = openai.model(model_embed)
		resp = llm.embed_many(texts, **kw)
	elif model_embed in ['all-mpnet-base-v2', 'multi-qa-distilbert-cos-v1', 'all-MiniLM-L6-v2']:
		model = SentenceTransformer(model_embed, device=DEVICE_CPU)
		# list of embeddings (texts count x embedding dim)
		embeddings = model.encode(
			sentences = texts,
			batch_size = 100,
			show_progress_bar = False,
			output_value = 'sentence_embedding', # Set to None, to get all output values ('sentence_embedding', 'token_embeddings')
			device = DEVICE_CPU,
		)
		resp = {}
		resp['model_embed'] = model_embed
		resp['vectors'] = embeddings.tolist()
	else:
		return

	resp['model_embed'] = model_embed
	return resp

# resp['vectors'] - lista embedding-a za svaki fragment teksta (npr. embedding za svaki paragraf)
# jedan openai embedding - lista dimenzije 1536
# list of embeddings (texts count x 1536)
# def embeddings(texts, **kw):
# 	model = kw.get('model','text-embedding-ada-002')
# 	llm = openai.model(model)
# 	resp = llm.embed_many(texts, **kw)
# 	resp['model'] = model
# 	return resp


# def embedding_local(text, **kw):
# 	model = SentenceTransformer(CHECKPOINT, device=DEVICE)
# 	# list of embeddings (texts count x embedding dim)
# 	embeddings = model.encode(
# 		sentences = text,
# 		batch_size = 100,
# 		show_progress_bar = False,
# 		output_value = 'sentence_embedding', # Set to None, to get all output values ('sentence_embedding', 'token_embeddings')
# 		device = DEVICE,
# 	)
# 	resp = {}
# 	resp['model'] = CHECKPOINT
# 	resp['vector'] = embeddings.tolist()
# 	# resp['usage'] = []

# 	return resp


# def embeddings_local(texts, **kw):
# 	model = SentenceTransformer(CHECKPOINT, device=DEVICE)
# 	# list of embeddings (texts count x embedding dim)
# 	embeddings = model.encode(
# 		sentences = texts,
# 		batch_size = 100,
# 		show_progress_bar = False,
# 		output_value = 'sentence_embedding', # Set to None, to get all output values ('sentence_embedding', 'token_embeddings')
# 		device = DEVICE,
# 	)
# 	resp = {}
# 	resp['model'] = CHECKPOINT
# 	resp['vectors'] = embeddings.tolist()
# 	# resp['usage'] = []
# 	return resp


tokenizer_model = openai.model('text-davinci-003')
def get_token_count(text):
	return tokenizer_model.token_count(text)

def stats_callback(out, resp, self):
	model = self.config['model']
	usage = resp['usage']
	usage['call_cnt'] = 1
	usage_stats.incr(f'usage:v4:[date]:[user]', {f'{k}:{model}':v for k,v in usage.items()})
	usage_stats.incr(f'hourly:v4:[date]',       {f'{k}:{model}:[hour]':v for k,v in usage.items()})

def get_community_usage_cost():
	data = usage_stats.get(f'usage:v4:[date]:{DEFAULT_USER}')
	used = 0.0
	used += 0.02   * data.get('total_tokens:text-davinci-003',0) / 1000
	used += 0.002  * data.get('total_tokens:text-curie-001',0) / 1000
	used += 0.002  * data.get('total_tokens:gpt-3.5-turbo',0) / 1000
	used += 0.0004 * data.get('total_tokens:text-embedding-ada-002',0) / 1000
	return used
