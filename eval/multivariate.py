#from anndata import AnnData
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from typing import Callable, Optional, TypeVar, Union, List
import networkx as nx
from scipy.spatial import distance
from scipy.stats import kstest
#from gprofiler import GProfiler
import itertools
warnings.filterwarnings("ignore")


def enrichr_to_gene_graph(db_path: str, perturbations: np.ndarray) -> nx.Graph:
    data = open(db_path).readlines()
    set_names = [x.split("\t\t")[0] for x in data]
    set_genes = [x.split("\t\t")[1].split("\t") for x in data]
    gene_sets = {}
    filt_set_genes = []
    for i, genes in enumerate(set_genes):
        genes.remove("\n")
        new_genes = [x for x in genes if x in perturbations]
        filt_set_genes.append(new_genes)
        gene_sets[set_names[i]] = new_genes
    merged = list(itertools.chain(*set_genes))
    unique_genes = set(merged)
    gene_graph = nx.Graph()
    gene_dict = {x: set() for x in unique_genes if x in perturbations}
    for genes in filt_set_genes:
        filt_genes = [x for x in genes if x in perturbations]
        for gene in filt_genes:
            gene_dict[gene].update(genes)  # add all genes as edges
            gene_dict[gene].remove(gene)  # remove self
    for gene in gene_dict:
        for connected in gene_dict[gene]:
            gene_graph.add_edge(gene, connected)
    return gene_graph, gene_sets


def corum_categories_to_gene_graphs(fcg_file: str, perturbations: np.ndarray):
    fcg = pd.read_csv(fcg_file, sep="\t")
    corum_categories = list(set(fcg.category_name.values))
    cat2genes = {}
    gene_graphs = []
    for _cat in corum_categories:
        genes = fcg[fcg.category_name == _cat]["subunits(Gene name)"].values
        unique_genes = set()
        all_genes = []
        for row in genes:
            _genes = row.split(";")
            for _gene in _genes:
                if _gene in perturbations:
                    all_genes.append(_gene)
                    unique_genes.add(_gene)
        gene_graph = nx.Graph()
        gene_dict = {x: set() for x in unique_genes if x in perturbations}
        for gene in all_genes:
            gene_dict[gene].update(all_genes)  # add all genes as edges
            gene_dict[gene].remove(gene)  # remove self
        for gene in gene_dict:
            for connected in gene_dict[gene]:
                gene_graph.add_edge(gene, connected)
        gene_graphs.append({"name": _cat, "graph": gene_graph})
    for graph in gene_graphs:
        print(graph["name"], len(graph["graph"].nodes))
    return gene_graphs


def find_children(term1, go, go_term_set={}, ret=False):
    for term2 in term1.children:
        go_term_set.update({term2})
        # Recurse on term to find all children
        find_children(term2, go, go_term_set)
    if(ret):
        return go_term_set

    
def go_terms_from_category(go, category):
    level_set = set()
    find_children(go[category], go, level_set)
    level_terms = []
    for term in list(level_set):
        level_terms.append(term.id)
        for child in term.children:
            level_terms.append(child.id)
        for parent in term.parents:
            level_terms.append(parent.id)
    return level_terms


def go_to_hierarchical_gene_graphs(db_path: str, ontology_path: str, perturbations):
    print("Reading in Gene Ontology..")
    go = obo_parser.GODag(ontology_path)
    data = open(db_path).readlines()
    set_names = [x.split("\t\t")[0] for x in data]
    set_genes = [x.split("\t\t")[1].split("\t") for x in data]
    go_terms = ["GO:" + x.split("(GO:")[1].strip(")") for x in set_names]
    go_terms_to_genes = {go_terms[i]: set_genes[i] for i in range(len(set_names))}
    
    print("Finding terms that are children of all level1 categories..")
    level1_categories = []
    for term in go:
        if go[term].level == 1:
            level1_categories.append(term)
    level1_to_go_terms = {x: [] for x in level1_categories}
    for _category in level1_categories:
        level1_to_go_terms[_category] = go_terms_from_category(go, _category)
    
    print("Filtering to terms that are in gene set db..")
    for _category in level1_categories:
        filt_terms = []
        for term in level1_to_go_terms[_category]:
            if term in go_terms_to_genes:
                filt_terms.append(term)
        level1_to_go_terms[_category] = filt_terms
    
    print("Creating gene graphs..")
    gene_graphs = []
    for i, _category in enumerate(level1_to_go_terms):
        set_genes = [go_terms_to_genes[x] for x in level1_to_go_terms[_category]]
        filt_set_genes = []
        for i, genes in enumerate(set_genes):
            #genes.remove("\n")
            new_genes = [x for x in genes if x in perturbations]
            filt_set_genes.append(new_genes)
        merged = list(itertools.chain(*set_genes))
        unique_genes = set(merged)
        gene_graph = nx.Graph()
        gene_dict = {x: set() for x in unique_genes if x in perturbations}
        for genes in filt_set_genes:
            filt_genes = [x for x in genes if x in perturbations]
            for gene in filt_genes:
                gene_dict[gene].update(genes)  # add all genes as edges
                gene_dict[gene].remove(gene)  # remove self
        for gene in gene_dict:
            for connected in gene_dict[gene]:
                gene_graph.add_edge(gene, connected)
        print(str(i) + "/" + str(len(level1_to_go_terms)), len(gene_graph.nodes))
        gene_graphs.append((_category, gene_graph))
    
    return gene_graphs


def recall_at_k(gene_graph: nx.Graph, nn_idxs: np.ndarray):
    """
    Calculates enrichment of gene-gene relationships from a DB
    in each embedding's kNN neighborhood. Recall@K.
    """
    genes = list(genes.nodes)
    intersection_length = 0
    num_edges = 0
    for i, nn_idx in enumerate(nn_idxs):
        identity_gene = genes[i]
        if identity_gene in gene_graph:
            edges = [x for x in list(gene_graph[identity_gene])]
            if len(edges) == 0:
                continue
            neighbor_genes = list(genes[nn_idx])
            cnt = 0
            for gene in edges:
                if gene in neighbor_genes:
                    cnt += 1
            intersection_length += cnt
            num_edges += len(edges)
    return intersection_length / float(num_edges)

def precision_at_k(gene_graph: nx.Graph, nn_idxs: np.ndarray):
    """
    Calculates enrichment of gene-gene relationships from a DB
    in each embedding's kNN neighborhood. Precision@K.
    """
    genes = list(genes.nodes)
    intersection_length = 0
    num_edges = 0
    for i, nn_idx in enumerate(nn_idxs):
        identity_gene = genes[i]
        if identity_gene in gene_graph:
            edges = [x for x in list(gene_graph[identity_gene])]
            if len(edges) == 0:
                continue
            neighbor_genes = list(genes[nn_idx])
            cnt = 0
            for gene in edges:
                if gene in neighbor_genes:
                    cnt += 1
            intersection_length += cnt
            num_edges += len(nn_idx)
    return intersection_length / float(num_edges)


def calculate_cluster_ks(gene_graph: nx.Graph, embeddings: np.ndarray, gene_sets: dict):
    cluster_sim = []
    random_sim = []
    gene_order = list(gene_graph.nodes)
    for set_name in gene_sets:
        if len(gene_sets[set_name]) > 0:
            genes = gene_sets[set_name]
            genes = [x for x in genes if gene_graph.has_node(x)]
            if len(genes) == 0:
                continue
            set_embeddings = embeddings[[gene_order.index(gene) for gene in genes], :]
            rand_genes = np.random.choice(gene_order, len(genes)) # smooth with constant oversampling + sample without replacement
            rand_embeddings = embeddings[
                [gene_order.index(gene) for gene in rand_genes], :
            ]
            set_pairwise_sim = cosine_similarity(set_embeddings)
            set_pairwise_sim = set_pairwise_sim[
                ~np.eye(set_pairwise_sim.shape[0], dtype=bool)
            ].reshape(set_pairwise_sim.shape[0], -1)
            rand_pairwise_sim = cosine_similarity(rand_embeddings)
            rand_pairwise_sim = rand_pairwise_sim[
                ~np.eye(rand_pairwise_sim.shape[0], dtype=bool)
            ].reshape(rand_pairwise_sim.shape[0], -1)
            if len(set_pairwise_sim) == 0 or len(rand_pairwise_sim) == 0:
                continue
            cluster_sim.append(set_pairwise_sim.mean())
            random_sim.append(rand_pairwise_sim.mean())
    ks_results = kstest(cluster_sim, random_sim)
    ks_inputs = {}
    ks_inputs["positive"] = cluster_sim
    ks_inputs["control"] = random_sim
    return ks_results, ks_inputs


def create_preds_and_labels(
    gene_graph, gene, dist, top_sim, bottom_sim, two_sided=True
):
    """
    Create labels and predictions for link prediction evaluation.
    Assumes that the order of dist matches list(gene_graph.nodes),
    which should always hold true if the embeddings are created using
    list(gene_graph.nodes) before calling the evaluation function.
    """
    nodes = list(gene_graph.nodes)
    gene_index = nodes.index(gene)
    # create ground truth labels
    labels = [1 if gene_graph.has_edge(gene, node) else 0 for node in nodes]
    # threshold the predicted distances
    preds = dist[nodes.index(gene), :]
    if two_sided:
        preds = [1 if ((dist_ <= top_sim) or (dist_ >= bottom_sim)) else 0 for dist_ in preds]
    else:
        preds = [1 if (dist_ <= top_sim) else 0 for dist_ in preds]
    preds = list(preds)
    # remove self-edges
    labels.pop(gene_index)
    preds.pop(gene_index)
    return labels, preds


def calculate_metrics(
    gene_graph: nx.Graph,
    dist: np.ndarray,
    cutoff_percentile=95,
    two_sided=True,
):
    """
    Calculates recall of gene-gene relationships from a DB
    in each embedding's neighborhood using a global formulation
    based on cosine similarity cutoff.
    """
    flattened_dist = dist.flatten()
    top_sim = np.percentile(flattened_dist, 100 - cutoff_percentile)
    bottom_sim = np.percentile(flattened_dist, cutoff_percentile)
    all_labels = []
    all_preds = []
    for gene in gene_graph:
        labels, preds = create_preds_and_labels(
            gene_graph, gene, dist, top_sim, bottom_sim, two_sided
        )
        all_labels += labels
        all_preds += preds
    if sum(all_preds) == 0:
        metrics = {}
        metrics["recall"] = 0.0
        return metrics
    metrics = {}
    metrics["recall"] = recall_score(all_labels, all_preds)
    metrics["precision"] = precision_score(all_labels, all_preds)
    metrics["f1_score"] = f1_score(all_labels, all_preds)
    tnr = recall_score(all_labels, all_preds, pos_label = 0)
    metrics["TPR"] = metrics["recall"]
    metrics["FPR"] = 1 - tnr
    return metrics


def interpret_embeddings(embeddings: np.ndarray, genes: list, cluster_assignment=None):
    if cluster_assignment is None:
        clustering_model = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.7
        )
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        clustering_model.fit(norm_embeddings)
        cluster_assignment = clustering_model.labels_
    clustered_genes = {}
    for gene_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_genes:
            clustered_genes[cluster_id] = []
        clustered_genes[cluster_id].append(genes[gene_id])
    cluster_enrichment = {}
    gp = GProfiler(return_dataframe=True)
    for cluster_id in clustered_genes:
        output = gp.profile(
            organism="hsapiens", query=clustered_genes[cluster_id], no_evidences=True
        )
        cluster_enrichment[cluster_id] = output
        print("Running enrichment for cluster: " + str(cluster_id))
    return cluster_enrichment


def evaluate_go(
    embeddings_obj,
    gene_graphs: List[nx.Graph],
    percentile = 95,
):   
    go = obo_parser.GODag("go-basic.obo")
    for i, gene_graph in enumerate(gene_graphs):
        if len(gene_graph[1].nodes) > 0: 
            print(len(gene_graph[1].nodes))
            embeddings = embeddings_obj.get_embeddings_from_list(list(gene_graph[1].nodes), agg=True)
            dist = distance.cdist(embeddings, embeddings, "cosine")
            metrics = calculate_metrics(
                gene_graph[1], dist, cutoff_percentile=percentile, two_sided=True
            )
            print(go[gene_graph[0]].name, " - recall @ 5%: " + str(metrics["recall"]))
            
            
def evaluate_corum_groups(
    embeddings_obj,
    gene_graphs: List[nx.Graph],
    percentile = 85,
):   
    for i, gene_graph in enumerate(gene_graphs):
        _name = gene_graph["name"]
        _graph = gene_graph["graph"]
        if len(_graph.nodes) > 0: 
            print(len(_graph.nodes))
            embeddings = embeddings_obj.get_embeddings_from_list(list(_graph.nodes), agg=True)
            dist = distance.cdist(embeddings, embeddings, "cosine")
            metrics = calculate_metrics(
                _graph, dist, cutoff_percentile=percentile, two_sided=True
            )
            print(_name, " - recall @ 5%: " + str(metrics["recall"]))

            
def evaluate(
    embeddings: np.ndarray,
    gene_graph: nx.Graph,
    gene_sets: dict,
    percentile_range=[80, 100],
    dist: np.ndarray=None,
):
    print("Performing Kolmogorov-Smirnov test on gene set cosine similarity..")
    ks_results, ks_inputs = calculate_cluster_ks(gene_graph, embeddings, gene_sets)
    print(
        "KS statistic: "
        + str(ks_results.statistic)
        + ", p-value: "
        + str(ks_results.pvalue)
    )
    bins = np.linspace(-1, 1, 100)
    plt.hist(ks_inputs["positive"], bins, alpha=0.5, label="Gene Sets CosineSim")
    plt.hist(ks_inputs["control"], bins, alpha=0.5, label="Random CosineSim")
    plt.title("KS plot comparing embedding similarity of gene sets vs random sets")
    plt.xlabel("Cosine Similarity")
    plt.legend(loc="upper left")
    plt.show()
    print("Computing distance matrix for embeddings..")
    if dist is None:
        dist = distance.cdist(embeddings, embeddings, "cosine")
    percentiles = np.arange(percentile_range[0], percentile_range[1], 1)
    tprs = []
    fprs = []
    precisions = []
    recalls = []
    f1_scores = []
    all_metrics = {}
    for percentile in percentiles:
        # print(
        #     "Computing link prediction metrics for cutoff percentile="
        #     + str(percentile)
        #     + "%"
        # )
        metrics = calculate_metrics(
            gene_graph, dist, cutoff_percentile=percentile, two_sided=True
        )
        tprs.append(metrics["TPR"])
        fprs.append(metrics["FPR"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1_scores.append(metrics["f1_score"])
        all_metrics[percentile] = metrics
    print("Finished prediction sweep over percentiles.")
    report = {}
    report["roc_auc"] = auc(fprs, tprs)
    report["auprc"] = auc(recalls, precisions)
    report["avg_f1"] = np.mean(f1_scores)
    report["metrics_by_percentile"] = all_metrics
    report["ks"] = ks_results
    return report, dist


class Embeddings:
    def get_embeddings_for(
        self, perturbation, batch=None, agg=False, centering=False
    ) -> np.ndarray:
        raise NotImplementedError()
    
    def get_embeddings_from_list(
        self, perturbations, agg = False, centering = False
    ) -> np.ndarray:
        raise NotImplementedError() 
    
    @property
    def perturbations(self) -> list:
        raise NotImplementedError()

    def perturbations_in_batch(
        self, batch, return_counts: bool = False
    ) -> Union[list, List[int]]:
        raise NotImplementedError()

    def random_batch_sample(
        self, batch, sample_size, perturbations=None, rng: np.random.Generator = None
    ):
        raise NotImplementedError()

    @property
    def batches(self):
        raise NotImplementedError()

    def perturbation_count(self, perturbation) -> int:
        raise NotImplementedError()

        
# class AnnDataEmbeddings(Embeddings):
#     def __init__(
#         self,
#         adata: AnnData,
#         embeddings_key: str,
#         perturbations_key: str,
#         batch_key: str,
#         ntc_name: str,
#     ):
#         self.adata = adata
#         self.identity_genes = np.array(self.adata.obs[perturbations_key].values)
#         self.embeddings_key = embeddings_key
#         self.perturbations_key = perturbations_key
#         self.batch_key = batch_key
#         self.ntc_name = ntc_name

#     @property
#     def _adata(self):
#         return self.adata
    
#     def _get_identity_genes(self):
#         return self.identity_genes

#     def _fetch_embeddings(self):
#         return self.adata.obsm[self.embeddings_key]

#     def _get_ntc_indices(self):
#         ntc_idxs = []
#         for i, _gene in enumerate(self.identity_genes):
#             if "NTC" in _gene:
#                 ntc_idxs.append(i)
#         return np.array(ntc_idxs)
#         #return np.where(self._get_identity_genes() == self.ntc_name)[0]

#     def _get_perturbation_indices(self, perturbation):
#         return np.where(self._get_identity_genes() == perturbation)[0]

#     def get_embeddings_for(self, perturbation, agg = True, centering = True) -> np.ndarray:
#         embeddings = self._fetch_embeddings()
#         ntc_indices = self._get_ntc_indices()
#         ntc_embedding = embeddings[ntc_indices, :].mean(0)
#         perturbation_indices = self._get_perturbation_indices(perturbation)
#         perturbation_embedding = embeddings[perturbation_indices, :]
#         if agg:
#             perturbation_embedding = np.mean(perturbation_embedding, axis=0)
#         if centering:
#             perturbation_embedding = perturbation_embedding - ntc_embedding
#         return perturbation_embedding.astype("float32")

#     @property
#     def perturbations(self) -> list:
#         return np.unique(self._get_identity_genes())

#     def perturbation_count(self, perturbation) -> int:
#         count = len(self._get_perturbation_indices(perturbation))
#         return count

#     @property
#     def batches(self):
#         return self.adata.obs[self.batch_key].unique()

#     def get_embeddings_from_list(self, genes, agg = True, centering = True):
#         embedding_dim = len(self.get_embeddings_for(self._get_identity_genes()[0]))
#         perturbation_embeddings = np.zeros((len(genes), embedding_dim))
#         for i, gene in enumerate(genes):
#             perturbation_embeddings[i, :] = self.get_embeddings_for(gene, agg, centering)
#         self.adata.uns["perturbation_embeddings"] = perturbation_embeddings
#         self.adata.uns["perturbations"] = genes
#         return perturbation_embeddings
