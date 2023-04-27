from ai_bricks.api import openai
from sentence_transformers import SentenceTransformer
import stats
import os
import streamlit as st

from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch

DEFAULT_USER = 'cb232f0236c29dc5' # community user
DEVICE_GPU = torch.device('cuda')
DEVICE_CPU = torch.device('cpu')

# in comments => max_seq_len - embed_dim
# CHECKPOINT = "all-mpnet-base-v2" # 384-768
# CHECKPOINT = "multi-qa-distilbert-cos-v1" # 512-768
# CHECKPOINT = "all-MiniLM-L6-v2" # 256-384

class StopOnTokens(StoppingCriteria):
	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		stop_ids = [50278, 50279, 50277, 1, 0]
		for stop_id in stop_ids:
			if input_ids[0][-1] == stop_id:
				return True
		return False


with st.spinner('Enabling GPU and Importing local models'):

	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
	print(f"Torch version: {torch.__version__}") 
	print(f"CUDA device count: {torch.cuda.device_count()}")
	print(f"Dev 0: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'}")
	print(f"Dev 1: {torch.cuda.get_device_properties(1).name if torch.cuda.is_available() else 'CPU'}")
	print(f"CUDA version: {torch.version.cuda}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# checkpoint = 'togethercomputer/GPT-NeoXT-Chat-Base-20B'
	# pipe = pipeline(
	# 	task="text-generation", 
	# 	model=checkpoint, 
	# 	tokenizer=checkpoint, 
	# 	use_fast=True,
	# 	# device=torch.device('cpu'),
	# 	device_map="auto", # auto  balanced_low_0
	# 	torch_dtype=torch.bfloat16, # torch.float16, torch.bfloat16, auto
	# 	model_kwargs = {
	# 		"temperature": 0,
	# 		# "load_in_8bit": True, # torch_dtype should be set to torch.float16
	# 		# "max_length": 4000,
	# 		# "max_new_tokens": 100,
	# 		# "offload_folder": "offload_folder/",
	# 		# "offload_state_dict": True,
	# 		# "max_memory": {0: "8GiB", 1: "22GiB", "cpu": "30GiB"},
	# 	}
	# )

	# clear cuda cache
	torch.cuda.empty_cache()


def format_response(response):
    # find first appearance of <human> in the response
    human_1 = response.find('<human>')
    # find second appearance of <human> in the response
    human_2 = response.find('<human>', human_1+1)
    # find first appearance of <bot> in the response
    bot_1 = response.find('<bot>')
    # extract the response
    return response[bot_1:human_2].replace('<bot>:', '')

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
		pipe = pipeline(
			task="text-generation", 
			model='togethercomputer/GPT-NeoXT-Chat-Base-20B', 
			tokenizer='togethercomputer/GPT-NeoXT-Chat-Base-20B', 
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
		response = pipe(f'''<human>: {text}\n<bot>:''', max_new_tokens=100)
		# print(response[0]['generated_text'])
		resp = {}
		resp['text'] = format_response(response[0]['generated_text'])
		resp['usage'] = {'tokens': 0, 'characters': 0, 'requests': 0}

	elif model in ['StableLM']:
		global model_hf, tokenizer_hf
		if 'model_hf' not in globals() or model_hf.config._name_or_path != 'StabilityAI/stablelm-tuned-alpha-7b':
			tokenizer_hf = AutoTokenizer.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b")
			model_hf = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b", device_map='sequential', torch_dtype=torch.float16)
		
		# StableLM Tuned should be used with prompts formatted to <|SYSTEM|>...<|USER|>...<|ASSISTANT|>... 
		system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
		- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
		- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
		- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
		- StableLM will refuse to participate in anything that could harm a human.
		"""
		user_prompt = text
		prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
		inputs = tokenizer_hf(prompt, return_tensors="pt").to(device)
		tokens = model_hf.generate(
			**inputs,
			max_new_tokens=1024,
			temperature=0.1,
			do_sample=True,
			stopping_criteria=StoppingCriteriaList([StopOnTokens()])
		)
		# print(tokenizer.decode(tokens[0], skip_special_tokens=True))
		resp = {}
		output = tokenizer_hf.decode(tokens[0], skip_special_tokens=True)
		resp['text'] = output[output.find('Answer:'):].replace('Answer:', '')
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
