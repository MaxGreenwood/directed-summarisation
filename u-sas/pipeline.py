from __future__ import division
import codecs
import os
import pickle
import json
import nltk
import collections
import operator
import sys
import copy
from matplotlib import pylab
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt
nltk.download('averaged_perceptron_tagger')

from read_data import *
from resolve_coref import *
from generate_document_graph import *
from tok_std_format_conversion import *
from directed_graph import Graph
from amr import AMR

def save_stories(stories,path=''):
	f = codecs.open(path,'w')
	for i in range(0,len(stories)):
		f.write(stories[i])
		f.write('\n')
	f.close()


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dataset', help="Name of dataset",
				type=str, default='amr_proxy')							
	parser.add_argument('--display', help="Path of the file containing AMRs of each sentence",
				type=bool, default=False)
	parser.add_argument('--read_only', help="Read files only (True) or run SGE (False)",
				type=bool, default=False)

	args = parser.parse_args(arguments)

	input_folder = f'{args.dataset}_inputs'
	output_folder = f'{args.dataset}_outputs'
	input_file = f'{input_folder}/amrs.txt'
	read_only = args.read_only
	# dataset = args.dataset

	'''
	'docs' is a list of 'documents', each 'document' is list a dictionary. Each dictionary contains
	information about a sentence. Each dicitonary has 'alignments', 'amr' etc. keys. Corresponding
	to each key we have the relevant information like the 'amr', 'text', 'alignment' etc.
	'''

	######################################################################
	# I. Reading in data
	######################################################################

	# Obtain stories target summaries from input file
	docs, target_summaries, stories = read_data(input_file) 		# [see read_data.py]

	# Write stories to .txt file
	save_stories(stories,f'{input_folder}/tok_stories.txt')
	# Write target summaries to .txt file
	with open(f'{output_folder}/target_summaries.txt','w') as f:
		for summary in target_summaries:
			f.write(tok_to_std_format_convertor(summary)+'\n')
	if read_only:
		return

	# Load IDF dictionary
	idf = {}
	with open(f'{input_folder}/idf_dict.json','rb') as f:
		idf = json.load(f) 

	# Load MFT dictionary
	mft = {}
	with open(f'{input_folder}/mft_dict.json','rb') as f:
		mft = json.load(f)

	# Load MFT embeddings dictionary
	mft_embeddings = {}
	with open(f'{input_folder}/mft_embeddings.json','rb') as f:
		mft_embeddings = json.load(f)

##############################################################
# II. SGE Algorithm
##############################################################

	# Initialise lists
	debug = False
	target_summaries_amrs = []			# list of target AMRs 
	predicted_summaries_amrs = []		# list of AMRs output using the SGE algorithm
	document_amrs = [] 					# list of document amrs formed after joining nodes and collapsing same entities etc.
	selected_sents = []

	# Loop over each document (i.e. article) which contains a list of dictionaries with sentence information
	for index_doc, doc in enumerate(docs):

		# TRIAL FOR SINGLE ARTICLE
		# if index_doc == 0:
			
			# Initialise lists for sentence AMRs
			current_doc_sent_amr_list = [] 						
			current_target_summary_sent_amr_list = [] 			

			# Loop over each sentence in the document
			for index_dict, dict_sentence in enumerate(doc):
				if dict_sentence['amr'] != []:
					if dict_sentence['tok'].strip()[-1] != '.': dict_sentence['tok'] = dict_sentence['tok'] + ' .'  # Add full stop as end token if not already there

					# Append AMR object to working summary list if sentence has 'summary' type
					if dict_sentence['snt-type'] == 'summary\n': # add \n for cnndm
						sent_amr = AMR(dict_sentence['amr'],
										amr_with_attributes=False,
										text=dict_sentence['tok'],
										alignments=dict_sentence['alignments'])
						current_target_summary_sent_amr_list.append(sent_amr)
						
					# Change dictionary AMR to AMR object and append to working document sentence list
					if dict_sentence['snt-type'] == 'body\n': # add \n for cnndm
						docs[index_doc][index_dict]['amr'] = AMR(dict_sentence['amr'],
															amr_with_attributes=False,
															text=dict_sentence['tok'],
															alignments=dict_sentence['alignments'])
						current_doc_sent_amr_list.append(docs[index_doc][index_dict]['amr'])

			# Merge the sentence AMRs in the working document sentence list to form a single AMR
			amr_as_list, document_text, document_alignments,var_to_sent = \
													merge_sentence_amrs(current_doc_sent_amr_list,debug=False) # [see generate_document_graph.py]
			new_document_amr = AMR(text_list=amr_as_list,
								text=document_text,
								alignments=document_alignments,
								amr_with_attributes=True,
								var_to_sent=var_to_sent)

			document_amrs.append(new_document_amr)			# Append whole document AMR to list
			target_summaries_amrs.append(current_target_summary_sent_amr_list)

			imp_doc = index_doc

			if index_doc == imp_doc:
				# Resolve coreferences: replace whole document AMR; find phrases and idf dictionary 
				document_amrs[index_doc], phrases, idf_vars = resolve_coref_doc_AMR(amr=document_amrs[index_doc], 		# [see resolve_coref.py]
										resolved=True,story=' '.join(document_amrs[index_doc].text),
										location_of_resolved_story=f'{input_folder}/coref.csv',								# replace coref csv
										location_of_story_in_file=index_doc,
										location_of_resolver='.',
										idf=idf,
										debug=False)

				cn_freq_dict,cn_sent_lists,cn_var_lists=document_amrs[index_doc].get_common_nouns(phrases=phrases) # [see amr.py]
				idf_vars = document_amrs[index_doc].get_idf_vars(idf_vars=idf_vars,idf=idf) # [see amr.py]

				#############################################################################

				mft_vars = document_amrs[index_doc].get_mft_vars(mft_vars={}, mft=mft, foundation='all')
				mft_embed_vars = document_amrs[index_doc].get_mft_embed_vars(mft_embed_vars={}, mft_embeddings=mft_embeddings[str(index_doc)], threshold=0.4913, foundation='authority')

				###############################################################################

				# # range equal to 4% of the number of nodes in the document AMR
				current_summary_nodes = []
				for target_summary_amr in current_target_summary_sent_amr_list:
					current_summary_nodes.extend(target_summary_amr.get_nodes() )
				num_summary_nodes = len(current_summary_nodes)
				range_num_nodes = int((len(document_amrs[index_doc].get_nodes())*4)/100)


				# Get concept relations from OpenIE output and assign to AMR
				document_amrs[index_doc].get_concept_relation_list(story_index=index_doc,debug=False)

				pr = document_amrs[index_doc].directed_graph.rank_sent_in_degree()
				ranks, weights, _ = zip(*pr)

				# rank the nodes with the 'meta_nodes'
				pr = document_amrs[index_doc].directed_graph.rank_with_meta_nodes(var_freq_list=pr,
																				cn_freq_dict=cn_freq_dict,
																				cn_sent_lists=cn_sent_lists,
																				cn_var_dict=cn_var_lists)
				ranks, weights, _ = zip(*pr)

				# rank the nodes with idf
				pr = document_amrs[index_doc].directed_graph.add_idf_ranking(var_freq_list=pr,
																			default_idf=1,
																			idf_vars=idf_vars,
																			num_vars_to_add=10)
				ranks, weights, _ = zip(*pr)

				#####################################################################

				# Choose type of MFT ranking
				mft_ranking = 'neither'

				if mft_ranking == 'keywords':
					# rank the nodes with MFT keywords
					pr = document_amrs[index_doc].directed_graph.add_mft_ranking(var_freq_list=pr, 
																				mft_factor=20, 
																				mft_vars=mft_vars)
					ranks, weights, _ = zip(*pr)

				elif mft_ranking == 'embeddings':
					# rank the nodes with embedding similarities
					pr = document_amrs[index_doc].directed_graph.add_mft_embed_ranking(var_freq_list=pr, 
																						mft_embed_factor=20, 
																						mft_embed_vars=mft_embed_vars,
																						sent_first_ranking=True)
					ranks, weights, _ = zip(*pr)

				#####################################################################

				# if index_doc == 37:
				# 	for var in ranks[::-1][:10]:
				# 		story_indices = document_amrs[index_doc].var_to_index[var]
				# 		# for idx in story_indices:
				# 		# 	print(document_amrs[index_doc].nodes[idx])
				# 		for node in document_amrs[index_doc].amr:
				# 			if node['variable'] == var:
				# 				if "/" in node['text']:
				# 					print(node['text'])
				# # 	breakpoint()

        # Extract the summary AMR graph 
				new_graph = document_amrs[index_doc].directed_graph.construct_greedily_first(ranks=ranks,weights=weights,
								concept_relation_list=document_amrs[index_doc].concept_relation_list,
								use_true_sent_rank=False,num_nodes=num_summary_nodes*2,range_num_nodes=range_num_nodes)

				# Generate AMR from the graphical representation
				new_amr_graph = document_amrs[index_doc].get_AMR_from_directed_graph(sub_graph=new_graph)
				predicted_summaries_amrs.append([new_amr_graph])
				print(f'Article {index_doc + 1} done')

	with open(f'{output_folder}/eos_stories.txt','w') as f:
		for document_amr in document_amrs:
			f.write(' <eos> '.join(document_amr.text)+'\n')

	target_summaries_nodes = []
	for target_summary_amrs in target_summaries_amrs:
		current_summary_nodes = []
		for target_summary_amr in target_summary_amrs:
			# current_summary_nodes.extend(target_summary_amr.get_edge_tuples() )
			current_summary_nodes.extend(target_summary_amr.get_nodes() )
		target_summaries_nodes.append(current_summary_nodes)

	target_summary_lengths = [len(i) for i in target_summaries_nodes]
	document_lengths = [len(i.get_nodes()) for i in document_amrs]

	ratios = []
	for i in range(len(document_lengths)):
		ratios.append(float(target_summary_lengths[i]/document_lengths[i])*100)

	average_ratio = (float(sum(ratios)) / len(ratios))
	deviations = [abs(ratio - average_ratio) for ratio in ratios]

	mean_deviation = (float(sum(deviations)) / len(deviations))

	# average ratio in 'gold' dataset is 9%, and deviation is 4%
	print('average_ratio', average_ratio, 'mean_deviation', mean_deviation)

	with open(f'{output_folder}/target_summary_nodes.txt','w') as f6:
		for node_list in target_summaries_nodes:
			f6.write(' '.join([node for node in node_list]) + '\n')

	predicted_summaries_nodes = []
	for predicted_summary_amrs in predicted_summaries_amrs:
		current_summary_nodes = []
		for predicted_summary_amr in predicted_summary_amrs:
			# current_summary_nodes.extend(predicted_summary_amr.get_edge_tuples() )
			current_summary_nodes.extend(predicted_summary_amr.get_nodes() )
		predicted_summaries_nodes.append(current_summary_nodes)

	# Write predicted AMR nodes to file (for calculating F1)
	with open(f'{output_folder}/predicted_summary_nodes.txt','w') as f7:
		for node_list in predicted_summaries_nodes:
			f7.write(' '.join([node for node in node_list]) + '\n')

	# Write predicted AMR to file
	with open(f'{output_folder}/predicted_summary_amrs.txt','w') as f8:
		for predicted_summary_amrs in predicted_summaries_amrs:
			for predicted_summary_amr in predicted_summary_amrs:
				f8.write(str(predicted_summary_amr.text_list) + '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
