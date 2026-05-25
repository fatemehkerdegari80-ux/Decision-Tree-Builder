import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
import os


class Node:
    def __init__(self, feature=None, children=None, threshold=None, samples=None, impurity=None, right=None, left=None, value=None, class_counts=None):
        # TODO: Initialize necessary attributes.
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.samples = samples
        self.impurity = impurity
        self.class_counts = class_counts if class_counts is not None else {}
        self.value = value             # Category value or predicted class (for leaf)
        self.left = left               # Left child (optional mirror of children[0])
        self.right = right             # Right child (optional mirror of children[1])
        
        def is_leaf_node(self):
            """
            Convenience helper: a node is a leaf if it has no children.
            """
            return self.children is None and self.left is None and self.right is None

class DecisionTree():
    def __init__(self, mode="gini", max_depth=None, min_samples_split=None,
                 class_weight=None, max_features=None, min_impurity_decrease=None, min_samples_leaf=1):
        # TODO: Save all hyperparameters
        self.mode = mode
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        # TODO: Initialize root node
        self.root = None
        self.decision_threshold = 0.5

    def evaluate(self, X, y, dataset_name="Validation"):
        """
        Evaluate the model on a given dataset.
        Returns a dictionary with common metrics.
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1] if self.predict_proba(X).shape[1] > 1 else self.predict_proba(X)[:, 0]

        metrics = {
            "dataset": dataset_name,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='binary', zero_division=0),
            "recall": recall_score(y, y_pred, average='binary', zero_division=0),
            "f1": f1_score(y, y_pred, average='binary', zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }

        # چاپ نتایج
        print(f"\n📊 نتایج {dataset_name}:")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
        print(f"   confusion_matrix:\n{np.array(metrics['confusion_matrix'])}")

        return metrics

    def _compute_class_weights(self, y):
        """
        Compute a weight for each class to handle class imbalance.

        Parameters
        ----------
        y : 1D numpy array
            Array of class labels

        Returns
        -------
        weights : dict
            Mapping from class label to its weight
        """

        classes, counts = np.unique(y, return_counts=True)

        # If no weighting requested
        if self.class_weight is None:
            return {c: 1.0 for c in classes}

        # Balanced weighting
        if self.class_weight == "balanced":
            n_samples = len(y)
            n_classes = len(classes)

            weights = {}
            for c, count in zip(classes, counts):
                weights[c] = n_samples / (n_classes * count)

            return weights

        # Custom dictionary provided
        if isinstance(self.class_weight, dict):
            return self.class_weight

        raise ValueError("Invalid value for class_weight")

        # TODO:
        # Handle three cases based on class_weight:
        #
        # Case 1 — None:
        #   Every class gets equal weight (1.0)
        #
        # Case 2 — "balanced":
        #   Weight for each class should be inversely proportional
        #   to how frequently it appears in y.
        #   Think about what formula makes a rare class weigh more.
        #
        # Case 3 — dict:
        #   The user already provided the weights, just return them.

        # weights = ...
        # return weights

    def _weighted_gini(self, y):
        """
        Compute the (possibly weighted) Gini impurity of a label array.

        Parameters
        ----------
        y : 1D numpy array
            Array of class labels for a node

        Returns
        -------
        impurity : float
            Gini impurity value in [0, 1]
        """
        """
        Compute the (possibly weighted) Gini impurity of a label array.
        """
        # 1. Empty array edge case
        if len(y) == 0:
            return 0.0

        # 2. Count occurrences of each class
        classes, counts = np.unique(y, return_counts=True)

        # 3. Apply class weights if available
        if getattr(self, "_class_weights", None) is None:
            # Unweighted probabilities
            probs = counts / counts.sum()
        else:
            # Weighted counts: count_c * w_c
            weights = np.array([self._class_weights.get(c, 1.0) for c in classes])
            weighted_counts = counts * weights
            total = weighted_counts.sum()
            # Guard against degenerate case
            if total == 0:
                return 0.0
            probs = weighted_counts / total

        # 4. Gini impurity
        gini = 1.0 - np.sum(probs ** 2)
        return float(gini)
        # TODO:
        # 1. Handle the empty array edge case.
        #
        # 2. Count how many times each class appears.
        #
        # 3. If class weights are available,
        #    scale each count by its class weight before computing probabilities.
        #    Make sure probabilities still sum to 1.
        #
        # 4. Compute and return the Gini impurity:
        #    impurity = ...

    def _weighted_entropy(self, y):
        """
        Compute the entropy of a label array.

        Parameters
        ----------
        y : 1D numpy array
            Array of class labels for a node

        Returns
        -------
        entropy : float
            Entropy value in bits (>= 0)
        """
        """
        Compute the (possibly weighted) entropy of a label array (in bits).
        """
        # 1. Empty array edge case
        if len(y) == 0:
            return 0.0

        # 2. Count occurrences of each class
        classes, counts = np.unique(y, return_counts=True)

        # 3. Apply class weights if available (same as in _weighted_gini)
        if getattr(self, "_class_weights", None) is None:
            probs = counts / counts.sum()
        else:
            weights = np.array([self._class_weights.get(c, 1.0) for c in classes])
            weighted_counts = counts * weights
            total = weighted_counts.sum()
            if total == 0:
                return 0.0
            probs = weighted_counts / total

        # 4. Entropy in bits: H = -sum p log2 p, avoid log(0)
        # Filter out zero probabilities to avoid log2(0)
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
        return float(entropy)
        # TODO:
        # 1. Handle the empty array edge case.
        #
        # 2. Count how many times each class appears.
        #
        # 3. If class weights are available, apply them the same way
        #    you did in _weighted_gini.
        #
        # 4. Compute and return entropy:
        #    H = ...
        #    avoid log(0).

        # return ...

    def _information_gain(self, parent_impurity, children_impurities, children_samples):
        """
        Calculate the weighted information gain of a split.

        Parameters
        ----------
        parent_impurity : float
            Impurity of the node before splitting
        children_impurities : list of float
            Impurity of each child node after the split
        children_samples : list of int
            Number of samples in each child node

        Returns
        -------
        gain : float
            Information gain (parent impurity minus weighted child impurity)
        """
        """
        Calculate the weighted information gain of a split.
        """

        # total samples in children
        total_samples = np.sum(children_samples)

        # avoid division by zero
        if total_samples == 0:
            return 0.0

        # weighted child impurity
        weighted_child_impurity = 0.0
        for imp, samp in zip(children_impurities, children_samples):
            weighted_child_impurity += (samp / total_samples) * imp

        # information gain
        gain = parent_impurity - weighted_child_impurity

        return gain
        # TODO:
        # 1. Compute the total number of samples across all children.
        #    Guard against division by zero.
        #
        # 2. Compute the weighted average impurity of the children.
        #    Each child's impurity is weighted by its fraction of total samples.
        #
        # 3. Return: parent_impurity - weighted_child_impurity

        # return ...

    def _get_best_split(self, X, y):
        """
        Search over features and split points to find the split with
        the highest information gain.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for the current node
        y : 1D numpy array
            Labels for the current node

        Returns
        -------
        best_split : dict or None
            Dictionary with keys:
              - "feature"   : column name
              - "type"      : "numerical" or "categorical"
              - "threshold" : cut-off value (numerical splits only)
              - "value"     : category value (categorical splits only)
              - "gain"      : information gain of this split
            Returns None if no valid split is found.
        """
        """
        Find the best feature and threshold to split on, according to the chosen mode
        ('gini' or 'entropy' / 'information_gain').

        For this exercise we focus on 'gini', using Gini Split (lower is better).
        Returns a dict with keys at least: 'feature' and 'threshold'.
        If no valid split is found, returns None.
        """
        # Convert X to DataFrame if it's not already, so we have column names
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        n_samples, n_features = X.shape

        # Respect min_samples_split: if not enough samples, no split
        if n_samples < self.min_samples_split:
            return None

        # Compute parent impurity once (for information gain, if you also support it)
        if self.mode == "gini":
            parent_impurity = self._weighted_gini(y)
        elif self.mode in ("entropy", "information_gain"):
            parent_impurity = self._weighted_entropy(y)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        best_feature = None
        best_threshold = None

        # For Gini mode: we want to minimize Gini Split
        if self.mode == "gini":
            best_score = np.inf
        else:
            # information gain: maximize
            best_score = -np.inf

        # Loop over all features
        for feature in X.columns:
            values = X[feature].values

            # Get sorted unique values to compute candidate thresholds
            unique_vals = np.unique(values)
            if unique_vals.size <= 1:
                # Can't split if all values are the same
                continue

            # Candidate thresholds: midpoints between consecutive unique values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thr in thresholds:
                # Left: values <= thr, Right: values > thr
                left_mask = values <= thr
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                # Respect min_samples_leaf: both sides must have enough samples
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                # Compute split quality
                if self.mode == "gini":
                    # Gini Split: weighted sum of child impurities
                    g_left = self._weighted_gini(y_left)
                    g_right = self._weighted_gini(y_right)

                    w_left = len(y_left) / n_samples
                    w_right = len(y_right) / n_samples

                    split_score = w_left * g_left + w_right * g_right

                    # We want the **lowest** Gini Split
                    if split_score < best_score:
                        best_score = split_score
                        best_feature = feature
                        best_threshold = float(thr)

                else:
                    # Information gain: use entropy & information gain
                    e_left = self._weighted_entropy(y_left)
                    e_right = self._weighted_entropy(y_right)

                    w_left = len(y_left) / n_samples
                    w_right = len(y_right) / n_samples

                    children_impurity = w_left * e_left + w_right * e_right
                    info_gain = parent_impurity - children_impurity

                    if info_gain > best_score:
                        best_score = info_gain
                        best_feature = feature
                        best_threshold = float(thr)

        # If no valid split found, return None
        if best_feature is None:
            return None

        # Return a dictionary as required by the tests
        return {
            "feature": best_feature,
            "threshold": best_threshold,
            # Optionally include more info if your tree-building code uses it:
            "score": best_score,
            "impurity": parent_impurity,
        }

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for the current subset
        y : 1D numpy array
            Labels for the current subset
        depth : int
            Current depth in the tree

        Returns
        -------
        node : Node
            Root of the (sub)tree built from this subset
        """
        n_samples = len(y)
        unique_classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique_classes, counts))
        impurity_fn = self._weighted_gini if self.mode == "gini" else self._weighted_entropy

        # Current node impurity
        node_impurity = impurity_fn(y)

        # ----- Stopping conditions → create a leaf -----
        # 1) Too few samples to split
        if self.min_samples_split is not None and n_samples < self.min_samples_split:
            leaf_value = self._weighted_most_common_class(y)
            return Node(
                feature=None,
                threshold=None,
                samples=n_samples,
                impurity=node_impurity,
                value=leaf_value,
                class_counts=class_counts
            )

        # 2) Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_value = self._weighted_most_common_class(y)
            return Node(
                feature=None,
                threshold=None,
                samples=n_samples,
                impurity=node_impurity,
                value=leaf_value,
                class_counts=class_counts
            )

        # 3) Pure node (only one class)
        if len(unique_classes) == 1:
            leaf_value = unique_classes[0]
            return Node(
                feature=None,
                threshold=None,
                samples=n_samples,
                impurity=node_impurity,
                value=leaf_value,
                class_counts=class_counts
            )

        # ----- Try to find the best split -----
        best_split = self._get_best_split(X, y)

        # If no valid split found → leaf
        if best_split is None:
            leaf_value = self._weighted_most_common_class(y)
            return Node(
                feature=None,
                threshold=None,
                samples=n_samples,
                impurity=node_impurity,
                value=leaf_value,
                class_counts=class_counts
            )

        feature = best_split["feature"]
        threshold = best_split["threshold"]

        # Numerical split: left <= threshold, right > threshold
        values = X[feature].values
        left_mask = values <= threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # Extra safety: if a split leads to an empty side, make this a leaf
        if len(y_left) == 0 or len(y_right) == 0:
            leaf_value = self._weighted_most_common_class(y)
            return Node(
                feature=None,
                threshold=None,
                samples=n_samples,
                impurity=node_impurity,
                value=leaf_value,
                class_counts=class_counts
            )

        # ----- Recursively build subtrees -----
        left_child = self._build_tree(X_left, y_left, depth=depth + 1)
        right_child = self._build_tree(X_right, y_right, depth=depth + 1)

        # ----- Internal decision node -----
        node = Node(
            feature=feature,
            threshold=threshold,
            samples=n_samples,
            impurity=node_impurity,
            value=None,
            class_counts=class_counts,
            children=(left_child, right_child),
            left=left_child,
            right=right_child
        )
        return node

        # TODO:
        # Check stopping conditions. Return a leaf Node when ANY of these hold:
        #   - n_samples < self.min_samples_split
        #   - depth >= self.max_depth
        #   - all labels are the same (only one unique class)
        # A leaf node has no children; its value is the predicted class.
        # Use _weighted_most_common_class to pick the predicted class.

        # TODO:
        # Try to find the best split using _get_best_split.
        # If no valid split is found, also return a leaf node.

        # TODO:
        # Apply the split to divide X and y into left and right subsets.
        #   - Numerical split:   left where feature <= threshold, right otherwise
        #   - Categorical split: left where feature == value,     right otherwise

        # TODO:
        # Recursively call _build_tree on the left and right subsets
        # (increment depth by 1).
        # Return an internal Node that stores the split information,
        # the two child nodes, sample count, and impurity.

    def visualize_tree(self, filename="decision_tree", format="png"):
        """
        Advanced compact Graphviz visualization with:
        • gradient leaf coloring (probability of class 1)
        • predicted probabilities in leaves
        • impurity values (gini or entropy)
        • compact node layout
        • colored edges (green=True, red=False)
        • bold majority class in leaves
        """

        dot = Digraph(name="DecisionTree", format=format)
        dot.attr(rankdir="TB")                  # vertical layout
        dot.attr("node", shape="box", style="filled,rounded",
                fontname="Helvetica", fontsize="10",
                margin="0.15")                # compact boxes
        dot.attr("edge", fontname="Helvetica", fontsize="9")

        # ---- helper: color gradient for leaf nodes (blue → red) ----
        def prob_to_color(p):
            """
            p = probability of class 1 (0..1)
            Blue (low p) → Red (high p)
            """
            r = int(p * 255)
            b = int((1 - p) * 255)
            return f"#{r:02x}00{b:02x}"

        # ---- recursive builder ----
        def add_nodes_edges(node, node_id):
            if node is None:
                return

            # === Compute leaf statistics ===
            if node.value is not None:  # leaf
                total = sum(node.class_counts.values())
                if total == 0:
                    p1 = 0.0
                else:
                    p1 = node.class_counts.get(1, 0) / total

                # gradient color based on class‑1 probability
                color = prob_to_color(p1)

                majority_class = node.value
                bold_major = f"<b>{majority_class}</b>"

                label = f"""<
                    <b>Leaf</b><br/>
                    samples={node.samples}<br/>
                    value={bold_major}<br/>
                    P(1)={p1:.3f}<br/>
                    class_counts={node.class_counts}
                >"""

                dot.node(node_id, label, fillcolor=color)
                return

            # === internal split node ===
            impurity = node.impurity if node.impurity is not None else 0
            thresh = node.threshold

            label = f"""<
                <b>{node.feature}</b> ≤ {thresh:.4f}<br/>
                samples={node.samples}<br/>
                impurity={impurity:.4f}
            >"""

            dot.node(node_id, label, fillcolor="#e8f3ff")  # light blue

            # children
            left_child, right_child = node.children

            # --- left (True) edge ---
            left_id = f"{node_id}L"
            dot.edge(node_id, left_id, label="True", color="green")
            add_nodes_edges(left_child, left_id)

            # --- right (False) edge ---
            right_id = f"{node_id}R"
            dot.edge(node_id, right_id, label="False", color="red")
            add_nodes_edges(right_child, right_id)

        # start recursion
        add_nodes_edges(self.root, "0")

        output_path = dot.render(filename, cleanup=True)
        print(f"Enhanced Graphviz tree saved to: {output_path}")



    def _weighted_most_common_class(self, y):
        """
        Return the class label that has the highest (weighted) count.

        Parameters
        ----------
        y : 1D numpy array
            Array of class labels

        Returns
        -------
        label : int or str
            The predicted class for a leaf node
        """
        """
        Return the class label that has the highest (weighted) count.
        """
        # 1. Handle empty array edge case
        if len(y) == 0:
            return None

        # 2. Count occurrences of each class in y
        classes, counts = np.unique(y, return_counts=True)

        # 3. If no class weights → majority vote based on raw counts
        if getattr(self, "_class_weights", None) is None:
            # classes[np.argmax(counts)] = class with highest raw count
            return classes[np.argmax(counts)]

        # 4. Weighted majority vote
        #    Multiply each count by its class weight
        #    Use weight 1.0 if class not present in dictionary
        weights = np.array([self._class_weights.get(c, 1.0) for c in classes])
        weighted_counts = counts * weights

        # class with highest weighted count
        return classes[np.argmax(weighted_counts)]

    def fit(self, X_train, y_train):
        """
        Fit the decision tree to training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix
        y_train : array-like
            Training labels

        Returns
        -------
        self
        """
        y_train = pd.Series(y_train).astype(int).values

        self.classes_ = np.unique(y_train)

        self._class_weights = self._compute_class_weights(y_train)

        print(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        if self.class_weight is not None:
            print(f"Class weights: {self._class_weights}")

        self.root = self._build_tree(X_train.copy(), y_train.copy())
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for each sample.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        probas : np.ndarray of shape (n_samples, 2)
            Each row is [P(class=0), P(class=1)]
        """
        probas = []

        for _, row in X.iterrows():
            # 1. Traverse tree to leaf
            leaf = self._traverse_to_leaf(row, self.root)

            counts = leaf.class_counts
            total = sum(counts.values())

            # 2. Handle empty leaf
            if total == 0:
                p1 = 0.5
            else:
                p1 = counts.get(1, 0) / total

            # 3. Append probability pair
            probas.append([1 - p1, p1])

        return np.array(probas)
        # TODO:
        # For each row in X:
        #   1. Traverse the tree to the appropriate leaf using _traverse_to_leaf.
        #   2. Use the leaf's class_counts to estimate P(class=1).
        #      Guard against an empty leaf (return 0.5 in that case).
        #   3. Append [1 - p1, p1] to the output list.
        # Return as a numpy array.

    def _traverse_to_leaf(self, sample, node):
        """
        Walk the tree from a given node down to the leaf that corresponds
        to a single sample.

        Parameters
        ----------
        sample : pd.Series
            A single row from the feature matrix
        node : Node
            Starting node (usually self.root)

        Returns
        -------
        leaf : Node
            The leaf node reached by following the split rules
        """
            # 1. Leaf check
        if node.value is not None:
            return node

        # 2. Feature extraction
        feature_value = sample[node.feature]

        # 3. Numerical split handling
        if node.threshold is not None:
            if feature_value <= node.threshold:
                return self._traverse_to_leaf(sample, node.children[0])
            else:
                return self._traverse_to_leaf(sample, node.children[1])

        # 4. Categorical split handling
        else:
            if feature_value == node.value:
                return self._traverse_to_leaf(sample, node.children[0])
            else:
                return self._traverse_to_leaf(sample, node.children[1])
        # TODO:
        # Base case: if the node has no children, it is a leaf — return it.
        #
        # Recursive case:
        #   Read the feature value from sample using node.feature.
        #   If node.threshold is not None  → numerical split:
        #       go left  if feature_value <= threshold
        #       go right otherwise
        #   Else                           → categorical split:
        #       go left  if feature_value == node.value
        #       go right otherwise

        # return ...

    def predict(self, X, threshold=None):
        """
        Predict class labels for each sample.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        threshold : float or None
            Decision threshold for class 1.
            Values >= threshold are predicted as class 1.
            Defaults to self.decision_threshold (initially 0.5).
            Lower  → higher recall  for class 1, lower precision.
            Higher → lower  recall  for class 1, higher precision.

        Returns
        -------
        predictions : 1D numpy array of int
        """
        # Threshold selection
        if threshold is None:
            threshold = self.decision_threshold

        # Probability extraction
        probas = self.predict_proba(X)

        # Use probability of class 1
        # Assumes class 1 is in column index 1
        p1 = probas[:, 1]

        # Binary conversion
        predictions = (p1 >= threshold).astype(int)

        return predictions
        # TODO:
        # 1. Use predict_proba to get the probability of class 1 for each sample.
        # 2. Apply the threshold to convert probabilities to binary labels.


    def tune_threshold(self, X_val, y_val, metric='f1', target_class=1):
        """
        Search for the decision threshold that maximises a chosen metric
        on a validation set.

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation feature matrix
        y_val : array-like
            Validation labels
        metric : str
            Metric to optimise: 'f1', 'recall', 'precision', or 'balanced'
            'balanced' minimises the difference between FPR and FNR.
        target_class : int
            Class to optimise for (default 1)

        Returns
        -------
        best_threshold : float
        best_score : float
        """
        y_val = np.array(y_val)

        # Step 1: compute probabilities
        probas = self.predict_proba(X_val)

        # Find index of the target_class in self.classes_
        class_index = np.where(self.classes_ == target_class)[0][0]

        # Probabilities for the target class
        target_probs = probas[:, class_index]

        # Step 2: define candidate thresholds
        thresholds = np.linspace(0.1, 0.9, 81)

        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            # Step 3: convert probabilities to predictions
            # If p >= threshold → predict target_class, else 0 (negative class)
            y_pred = np.where(target_probs >= threshold, target_class, 0)

            # Step 4: evaluate according to the chosen metric
            if metric == 'precision':
                tp = np.sum((y_val == target_class) & (y_pred == target_class))
                fp = np.sum((y_val != target_class) & (y_pred == target_class))
                score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            elif metric == 'recall':
                tp = np.sum((y_val == target_class) & (y_pred == target_class))
                fn = np.sum((y_val == target_class) & (y_pred != target_class))
                score = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            elif metric == 'f1':
                tp = np.sum((y_val == target_class) & (y_pred == target_class))
                fp = np.sum((y_val != target_class) & (y_pred == target_class))
                fn = np.sum((y_val == target_class) & (y_pred != target_class))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            elif metric == 'balanced':
                tp = np.sum((y_val == target_class) & (y_pred == target_class))
                tn = np.sum((y_val != target_class) & (y_pred != target_class))
                fp = np.sum((y_val != target_class) & (y_pred == target_class))
                fn = np.sum((y_val == target_class) & (y_pred != target_class))

                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

                score = 1.0 - abs(fpr - fnr)

            else:
                raise ValueError("metric must be one of: 'f1', 'recall', 'precision', 'balanced'")

            # Step 5: track best threshold
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        # Step 6: store optimal threshold
        self.decision_threshold = best_threshold

        return best_threshold, best_score

        # TODO:
        # Iterate over each candidate threshold.
        # Convert probabilities to binary predictions using the threshold.
        # Compute the chosen metric:
        #   'f1'        → F1 score for target_class
        #   'recall'    → Recall for target_class
        #   'precision' → Precision for target_class
        #   'balanced'  → 1 - |FPR - FNR|  (use the confusion matrix)
        #                 FPR = FP / (FP + TN),  FNR = FN / (FN + TP)
        # Track the threshold that gives the highest score.
        # After the loop, store it in self.decision_threshold and print it.

        # return best_threshold, best_score

    def get_feature_importance(self):
        """
        Estimate feature importances as the total (weighted) impurity
        decrease each feature contributes across all splits.

        Returns
        -------
        importances : dict
            Mapping from feature name to normalised importance score,
            sorted in descending order.
        """

        from collections import defaultdict
        importances = defaultdict(float)

        def _calculate_importance(node):
            # 1. Stop at leaf nodes
            if node.children is None:
                return

            left_child, right_child = node.children
            n_left, n_right = left_child.samples, right_child.samples
            n_total = node.samples

            # Validate sample counts
            if n_total == 0:
                return

            # 2. Compute impurity decrease
            weighted_child_impurity = (n_left / n_total) * left_child.impurity + (n_right / n_total) * right_child.impurity
            decrease = node.impurity - weighted_child_impurity

            # 3. Weight decrease by node size and accumulate
            importances[node.feature] += decrease * n_total

            # 4. Recursively compute importance for subtrees
            _calculate_importance(left_child)
            _calculate_importance(right_child)

        # 5. Start computation from the root
        if self.root is not None:
            _calculate_importance(self.root)

        # 6. Normalize importances
        total_importance = sum(importances.values())
        if total_importance > 0:
            for f in importances:
                importances[f] /= total_importance

        # 7. Store and return sorted dictionary
        self._feature_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        return self._feature_importances
        

def heatmap_gini_filter_clean(df, target_col, threshold=0.005, sample_size=None,
                               min_group_size=30, random_state=42, show_plot=True):
    """
    Select features by measuring how much each one reduces Gini impurity,
    then visualise the results as a heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column
    target_col : str
        Name of the target column
    threshold : float
        Features whose normalised Gini reduction is below this value are removed
    sample_size : int or None
        If provided, randomly subsample the dataframe before processing
    min_group_size : int
        Groups smaller than this are skipped when computing per-value Gini
    random_state : int
        Random seed for reproducibility
    show_plot : bool
        Whether to display the heatmap

    Returns
    -------
    cleaned_df : pd.DataFrame
        DataFrame with low-impact features removed
    report : pd.DataFrame
        Table of features and their normalised Gini reduction scores
    """
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state).copy()

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int).values

    def gini(arr):
        # 1. Handle empty array
        if len(arr) == 0:
            return 0.0
        classes, counts = np.unique(arr, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    # Parent impurity before split
    gini_parent = gini(y)
    impacts = {}

    # ---- Compute ΔGini for each feature ----
    for feature in X.columns:
        col_values = X[feature].values
        unique_vals = np.unique(col_values)

        # Skip constant features
        if len(unique_vals) <= 1:
            continue

        gini_child = 0.0
        total_size = len(y)
        valid_groups = 0

        for val in unique_vals:
            mask = col_values == val
            group = y[mask]
            group_size = len(group)

            if group_size < min_group_size:
                continue

            gini_group = gini(group)
            gini_child += (group_size / total_size) * gini_group
            valid_groups += 1

        if valid_groups == 0:
            continue

        delta_norm = (gini_parent - gini_child) / (gini_parent + 1e-10)
        impacts[feature] = delta_norm

    # ---- Create report DataFrame ----
    report = pd.DataFrame(list(impacts.items()), columns=["Feature", "ΔGini_norm"])
    report.sort_values("ΔGini_norm", ascending=False, inplace=True)
    report.reset_index(drop=True, inplace=True)

    # Features below threshold
    remove_cols = report.loc[report["ΔGini_norm"] < threshold, "Feature"].tolist()

    cleaned_df = df.drop(columns=remove_cols).copy()

    # Reorder columns: target last
    cols = [c for c in cleaned_df.columns if c != target_col] + [target_col]
    cleaned_df = cleaned_df[cols]

    print(f"Removed {len(remove_cols)} low-impact features, kept {cleaned_df.shape[1]-1} predictors.")

    # ---- Plot heatmap ----
    if show_plot and not report.empty:
        plt.figure(figsize=(8, max(4, len(report) * 0.3)))
        sns.heatmap(
            report[["ΔGini_norm"]].set_index(report["Feature"]),
            annot=True, fmt=".3f", cmap="coolwarm", cbar=False
        )
        plt.title(f'Feature Gini Reduction Heatmap (threshold={threshold})')
        plt.xlabel('ΔGini_norm')
        plt.ylabel('Features')

        # Dashed horizontal line separating kept/removed features
        below_thresh_idx = np.sum(report["ΔGini_norm"] >= threshold)
        if 0 < below_thresh_idx < len(report):
            plt.axhline(y=below_thresh_idx, color="red", linestyle="--")

        plt.tight_layout()
        plt.show()

    return cleaned_df, report


