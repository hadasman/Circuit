import multiprocessing
from tqdm import tqdm
from collections import defaultdict

def worker_job(data):
	
	pre, single_pre_to_post = data[0], data[1]
	post_to_pre, all_posts = {}, []

	for post in single_pre_to_post: 
		post_to_pre[post] = {}
		post_to_pre[post][pre] = single_pre_to_post[post] 
		all_posts.append(post)

	return post_to_pre, all_posts

pool = multiprocessing.Pool()

# pres = [i for i in connectivity]
# single_pre_to_posts = [connectivity[pre] for pre in pres]

post_to_pre_array, all_posts = [], []
for i in list(tqdm(pool.imap(worker_job, list(connectivity.items())))):
	post_to_pre_array.append(i[0])
	all_posts.append(i[1])
all_posts = list(set([j for i in all_posts for j in i]))

d = {post: {} for post in all_posts}
for post_to_pre in post_to_pre_array:
	for post in post_to_pre:

		pre = list(post_to_pre[post].keys())
		assert len(pre)==1, 'Problem #1'
		pre = pre[0]

		assert pre not in d[post].keys(), 'Problem #2'
		d[post][pre] = post_to_pre[post][pre]