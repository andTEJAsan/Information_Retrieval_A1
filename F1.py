import sys
from collections import defaultdict
import json
result_file = sys.argv[1]
qrels = sys.argv[2]

print(f"here")
sys.stdout.flush()
relevant_docs = defaultdict(set)
precs = {}
recalls = {}
total_extracted = defaultdict(int)
total_relevant_extracted = defaultdict(int)
with open(qrels, 'r') as f:
	for line in f:
		data = json.loads(line)
		qid = data["query_id"]
		docid = data["doc_id"]
		relevance = float(data["relevance"])
		if(relevance > 0):
			relevant_docs[qid].add(docid)
# print(f"relevant_docs: {relevant_docs}")
with open(result_file, 'r') as f:
	# ignore first line
	f.readline()
	for line in f:
		qid, iteration, docid, relevance = line.split()
		if docid in relevant_docs[qid]:
			total_relevant_extracted[qid] += 1
		total_extracted[qid] += 1


f1 = {}
for qid in total_extracted.keys():
	precs[qid] = total_relevant_extracted[qid] / total_extracted[qid]
	# print(f"qid: {qid}, precs: {precs[qid]}")
	recalls[qid] = total_relevant_extracted[qid] / len(relevant_docs[qid])
	# print(f"qid: {qid}, recalls: {recalls[qid]}")
	if(precs[qid] + recalls[qid] == 0):
		f1[qid] = 0
	else:
		f1[qid] = 2 * precs[qid] * recalls[qid] / (precs[qid] + recalls[qid])
print(f"average precision: {sum(precs.values()) / len(precs)}")
print(f"average recall: {sum(recalls.values()) / len(recalls)}")
print(f"average f1: {sum(f1.values()) / len(f1)}")

