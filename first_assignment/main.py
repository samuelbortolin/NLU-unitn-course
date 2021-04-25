from __future__ import absolute_import, annotations

from typing import List, Dict, Union

import nltk
nltk.download("dependency_treebank")
from nltk.corpus import dependency_treebank
from nltk.parse.transitionparser import *
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.ensemble import GradientBoostingClassifier
import spacy
from spacy import Language
from spacy.tokens import Token, Doc, Span


spacy_nlp: Language = spacy.load("en_core_web_sm")


# Working with Dependency Graphs (Parses)

# 1. extract a path of dependency relations from the ROOT to a token
def extract_path_of_dependency_relations(sentence: str) -> List[List[str]]:
    if not isinstance(sentence, str):
        raise TypeError("You pass a `sentence` parameter of a wrong type")

    spacy_doc: Doc = spacy_nlp(sentence)  # parse the input sentence and get a Doc object of spaCy
    sentence_dependency_relations: List[List[str]] = []
    for token in spacy_doc:  # for each token, extract the path that is a list of dependency relations, where first element is ROOT
        token_dependency_relations: List[str] = [token.text, f"--{token.dep_}-->"]
        while token.dep_ != "ROOT":
            token: Token = token.head
            token_dependency_relations.append(token.text)
            token_dependency_relations.append(f"--{token.dep_}-->")

        token_dependency_relations.append("ROOT")
        token_dependency_relations.reverse()
        sentence_dependency_relations.append(token_dependency_relations)

    return sentence_dependency_relations


# 2. extract subtree of dependents given a token
def extract_dependents_subtree(sentence: str) -> List[List[str]]:
    if not isinstance(sentence, str):
        raise TypeError("You pass a `sentence` parameter of a wrong type")

    spacy_doc: Doc = spacy_nlp(sentence)  # parse the input sentence and get a Doc object of spaCy
    sentence_dependents_subtrees: List[List[str]] = []
    for token in spacy_doc:  # for each token, extract a subtree of its dependents as a list (ordered w.r.t. sentence order)
        token_dependents_subtree: List[str] = [subtree_token.text for subtree_token in token.subtree]
        sentence_dependents_subtrees.append(token_dependents_subtree)

    return sentence_dependents_subtrees


# 3. check if a given list of tokens (segment of a sentence) forms a subtree
def check_if_tokens_form_a_subtree(sentence: str, tokens: Union[Span, List[Token], List[str], str]) -> bool:
    if not isinstance(sentence, str):
        raise TypeError("You pass a `sentence` parameter of a wrong type")

    if isinstance(tokens, Span):
        tokens = [token.text for token in tokens]
    elif isinstance(tokens, str):
        tokens = [token.text for token in spacy_nlp(tokens)]
    elif not isinstance(tokens, list):
        raise TypeError("You pass a `tokens` parameter of a wrong type")

    for index, token in enumerate(tokens):
        if isinstance(token, Token):
            tokens[index] = token.text
        elif not isinstance(token, str):
            raise TypeError("You pass a `tokens` parameter with elements of a wrong type")

    sentence_dependents_subtrees: List[List[str]] = extract_dependents_subtree(sentence)
    if tokens in sentence_dependents_subtrees:  # providing as an input ordered list of words from a sentence, output True/False based on the sequence forming a subtree or not
        tokens_form_a_subtree: bool = True
    else:
        tokens_form_a_subtree: bool = False

    return tokens_form_a_subtree


# 4. identify head of a span, given its tokens
def identify_head_of_a_span(span: Union[Span, List[Token], List[str], str]) -> str:
    if isinstance(span, list):  # input is a sequence of words (not necessarily a sentence)
        for index, token in enumerate(span):
            if isinstance(token, Token):
                span[index] = token.text
            elif not isinstance(token, str):
                raise TypeError("You pass a `span` parameter with elements of a wrong type")

        span_string = TreebankWordDetokenizer().detokenize(span)
        spacy_doc: Doc = spacy_nlp(span_string)
        span = spacy_doc[:]
    elif isinstance(span, str):
        spacy_doc: Doc = spacy_nlp(span)
        span = spacy_doc[:]
    elif not isinstance(span, Span):
        raise TypeError("You pass a `span` parameter of a wrong type")

    root: Token = span.root  # output is the head of the span (single word)
    return root.text


# 5. extract sentence subject, direct object and indirect object spans
def extract_subj_dobj_iobj(sentence: str) -> Dict[str, List[str]]:
    if not isinstance(sentence, str):
        raise TypeError("You pass a `sentence` parameter of a wrong type")

    spacy_doc: Doc = spacy_nlp(sentence)  # parse the input sentence and get a Doc object of spaCy
    subj_dobj_iobj: Dict[str, List[str]] = dict({"subj": list(), "dobj": list(), "iobj": list()})  # output is dict of lists of words that form a span for subject, direct object, and indirect object (if present, otherwise empty)
    for token in spacy_doc:
        if token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ == "nsubj" or child.dep_ == "nsubjpass" or child.dep_ == "csubj" or child.dep_ == "csubjpass" or child.dep_ == "expl":
                    subj_dobj_iobj["subj"].extend([subtree_token.text for subtree_token in child.subtree])
                elif child.dep_ == "dobj":
                    subj_dobj_iobj["dobj"].extend([subtree_token.text for subtree_token in child.subtree])
                elif child.dep_ == "dative":  # in spaCy "dative" is used instead of "iobj" (that is deprecated)
                    subj_dobj_iobj["iobj"].extend([subtree_token.text for subtree_token in child.subtree])

            break

    return subj_dobj_iobj


if __name__ == "__main__":

    example_sentence: str = "I saw a man with a telescope, he was looking at the Moon."
    print(f"The example sentence used will be: `{example_sentence}`")
    print()

    print("1. extract a path of dependency relations from the ROOT to a token:")
    print(extract_path_of_dependency_relations(example_sentence))
    print()

    print("2. extract subtree of dependents given a token:")
    print(extract_dependents_subtree(example_sentence))
    print()

    example_span: Span = spacy_nlp(example_sentence)[2:7]
    wrong_span: Span = spacy_nlp(example_sentence)[5:8]

    print("3. check if a given list of tokens (segment of a sentence) forms a subtree:")
    print(f"Does this example span `{example_span}` (passing the span object) form a subtree? `{check_if_tokens_form_a_subtree(example_sentence, example_span)}`")
    print(f"Does this example span `'{example_span.text}'` (passing the span as string) form a subtree? `{check_if_tokens_form_a_subtree(example_sentence, example_span.text)}`")
    print(f"Does this example span `{[token for token in example_span]}` (passing the span as tokens objects) form a subtree? `{check_if_tokens_form_a_subtree(example_sentence, [token for token in example_span])}`")
    print(f"Does this example span `{[token.text for token in example_span]}` (passing the span as tokens strings) form a subtree? `{check_if_tokens_form_a_subtree(example_sentence, [token.text for token in example_span])}`")
    print(f"Does this wrong span `{wrong_span}` (passing the span object) form a subtree? `{check_if_tokens_form_a_subtree(example_sentence, wrong_span)}`")
    print(f"Does this wrong span `'{wrong_span.text}'` (passing the span as string) form a subtree? `{check_if_tokens_form_a_subtree(example_sentence, wrong_span.text)}`")
    print(f"Does this wrong span `{[token for token in wrong_span]}` (passing the span as tokens objects) form a subtree? `{check_if_tokens_form_a_subtree(example_sentence, [token for token in wrong_span])}`")
    print(f"Does this wrong span `{[token.text for token in wrong_span]}` (passing the span as tokens strings) form a subtree? `{check_if_tokens_form_a_subtree(example_sentence, [token.text for token in wrong_span])}`")
    print()

    print("4. identify head of a span, given its tokens:")
    print(f"The head of this example span `{example_span}` (passing the span object) is: `{identify_head_of_a_span(example_span)}`")
    print(f"The head of this example span `'{example_span.text}'` (passing the span as string): `{identify_head_of_a_span(example_span.text)}`")
    print(f"The head of this example span `{[token for token in example_span]}` (passing the span as tokens objects): `{identify_head_of_a_span([token for token in example_span])}`")
    print(f"The head of this example span `{[token.text for token in example_span]}` (passing the span as tokens strings): `{identify_head_of_a_span([token.text for token in example_span])}`")
    print()

    print("5. extract sentence subject, direct object and indirect object spans:")
    print(extract_subj_dobj_iobj(example_sentence))
    print()

    # Training Transition-Based Dependency Parser (Optional & Advanced)

    # Modify NLTK Transition parser ' s Configuration class to use better features
    class MyConfiguration(Configuration):

        def extract_features(self):
            result = []
            if len(self.stack) > 0:
                stack_idx0 = self.stack[len(self.stack) - 1]
                token = self._tokens[stack_idx0]
                if "head" in token and self._check_informative(token["head"]):
                    result.append("STK_0_HEAD_" + str(token["head"]).upper())
                if "lemma" in token and self._check_informative(token["lemma"]):
                    result.append("STK_0_LEMMA_" + token["lemma"].upper())
                if "tag" in token and self._check_informative(token["tag"]):
                    result.append("STK_0_POS_" + token["tag"].upper())
                if "rel" in token and self._check_informative(token["rel"]):
                    result.append("STK_0_REL_" + token["rel"].upper())
                if "deps" in token and token["deps"]:
                    for d in token["deps"]:
                        result.append("STK_0_DEP_" + str(d).upper())
                if "feats" in token and self._check_informative(token["feats"]):
                    feats = token["feats"].split("|")
                    for feat in feats:
                        result.append("STK_0_FEATS_" + feat.upper())
                if len(self.stack) > 1:
                    stack_idx1 = self.stack[len(self.stack) - 2]
                    token = self._tokens[stack_idx1]
                    if "head" in token and self._check_informative(token["head"]):
                        result.append("STK_1_HEAD_" + str(token["head"]).upper())
                    if "lemma" in token and self._check_informative(token["lemma"]):
                        result.append("STK_1_LEMMA_" + token["lemma"].upper())
                    if "tag" in token and self._check_informative(token["tag"]):
                        result.append("STK_1_POS_" + token["tag"].upper())
                    if "rel" in token and self._check_informative(token["rel"]):
                        result.append("STK_1_REL_" + token["rel"].upper())
                    if "deps" in token and token["deps"]:
                        for d in token["deps"]:
                            result.append("STK_1_DEP_" + str(d).upper())
                    if "feats" in token and self._check_informative(token["feats"]):
                        feats = token["feats"].split("|")
                        for feat in feats:
                            result.append("STK_1_FEATS_" + feat.upper())
                if len(self.stack) > 2:
                    stack_idx2 = self.stack[len(self.stack) - 3]
                    token = self._tokens[stack_idx2]
                    if self._check_informative(token["tag"]):
                        result.append("STK_2_POS_" + token["tag"].upper())
                if len(self.stack) > 3:
                    stack_idx3 = self.stack[len(self.stack) - 4]
                    token = self._tokens[stack_idx3]
                    if self._check_informative(token["tag"]):
                        result.append("STK_3_POS_" + token["tag"].upper())

                left_most = 1000000
                right_most = -1
                dep_left_most = ""
                dep_right_most = ""
                for (wi, r, wj) in self.arcs:
                    if wi == stack_idx0:
                        if (wj > wi) and (wj > right_most):
                            right_most = wj
                            dep_right_most = r
                        if (wj < wi) and (wj < left_most):
                            left_most = wj
                            dep_left_most = r
                if self._check_informative(dep_left_most):
                    result.append("STK_0_LDEP_" + dep_left_most.upper())
                if self._check_informative(dep_right_most):
                    result.append("STK_0_RDEP_" + dep_right_most.upper())

            if len(self.buffer) > 0:
                buffer_idx0 = self.buffer[0]
                token = self._tokens[buffer_idx0]
                if "head" in token and self._check_informative(token["head"]):
                    result.append("BUF_0_HEAD_" + str(token["head"]).upper())
                if "lemma" in token and self._check_informative(token["lemma"]):
                    result.append("BUF_0_LEMMA_" + token["lemma"].upper())
                if "tag" in token and self._check_informative(token["tag"]):
                    result.append("BUF_0_POS_" + token["tag"].upper())
                if "rel" in token and self._check_informative(token["rel"]):
                    result.append("BUF_0_REL_" + token["rel"].upper())
                if "deps" in token and token["deps"]:
                    for d in token["deps"]:
                        result.append("BUF_0_DEP_" + str(d).upper())
                if "feats" in token and self._check_informative(token["feats"]):
                    feats = token["feats"].split("|")
                    for feat in feats:
                        result.append("BUF_0_FEATS_" + feat.upper())
                if len(self.buffer) > 1:
                    buffer_idx1 = self.buffer[1]
                    token = self._tokens[buffer_idx1]
                    if "head" in token and self._check_informative(token["head"]):
                        result.append("BUF_1_HEAD_" + str(token["head"]).upper())
                    if "lemma" in token and self._check_informative(token["lemma"]):
                        result.append("BUF_1_LEMMA_" + token["lemma"].upper())
                    if "tag" in token and self._check_informative(token["tag"]):
                        result.append("BUF_1_POS_" + token["tag"].upper())
                    if "rel" in token and self._check_informative(token["rel"]):
                        result.append("BUF_1_REL_" + token["rel"].upper())
                    if "deps" in token and token["deps"]:
                        for d in token["deps"]:
                            result.append("BUF_1_DEP_" + str(d).upper())
                    if "feats" in token and self._check_informative(token["feats"]):
                        feats = token["feats"].split("|")
                        for feat in feats:
                            result.append("BUF_1_FEATS_" + feat.upper())
                if len(self.buffer) > 2:
                    buffer_idx2 = self.buffer[2]
                    token = self._tokens[buffer_idx2]
                    if self._check_informative(token["tag"]):
                        result.append("BUF_2_POS_" + token["tag"].upper())
                if len(self.buffer) > 3:
                    buffer_idx3 = self.buffer[3]
                    token = self._tokens[buffer_idx3]
                    if self._check_informative(token["tag"]):
                        result.append("BUF_3_POS_" + token["tag"].upper())

                left_most = 1000000
                right_most = -1
                dep_left_most = ""
                dep_right_most = ""
                for (wi, r, wj) in self.arcs:
                    if wi == buffer_idx0:
                        if (wj > wi) and (wj > right_most):
                            right_most = wj
                            dep_right_most = r
                        if (wj < wi) and (wj < left_most):
                            left_most = wj
                            dep_left_most = r
                if self._check_informative(dep_left_most):
                    result.append("BUF_0_LDEP_" + dep_left_most.upper())
                if self._check_informative(dep_right_most):
                    result.append("BUF_0_RDEP_" + dep_right_most.upper())

            return result


    class MyTransitionParser(TransitionParser):

        def _create_training_examples_arc_std(self, depgraphs, input_file):
            operation = Transition(self.ARC_STANDARD)
            count_proj = 0
            training_seq = []

            for depgraph in depgraphs:
                if not self._is_projective(depgraph):
                    continue

                count_proj += 1
                conf = MyConfiguration(depgraph)
                while len(conf.buffer) > 0:
                    b0 = conf.buffer[0]
                    features = conf.extract_features()
                    binary_features = self._convert_to_binary_features(features)

                    if len(conf.stack) > 0:
                        s0 = conf.stack[len(conf.stack) - 1]
                        # Left-arc operation
                        rel = self._get_dep_relation(b0, s0, depgraph)
                        if rel is not None:
                            key = Transition.LEFT_ARC + ":" + rel
                            self._write_to_file(key, binary_features, input_file)
                            operation.left_arc(conf, rel)
                            training_seq.append(key)
                            continue

                        # Right-arc operation
                        rel = self._get_dep_relation(s0, b0, depgraph)
                        if rel is not None:
                            precondition = True
                            # Get the max-index of buffer
                            maxID = conf._max_address

                            for w in range(maxID + 1):
                                if w != b0:
                                    relw = self._get_dep_relation(b0, w, depgraph)
                                    if relw is not None:
                                        if (b0, relw, w) not in conf.arcs:
                                            precondition = False

                            if precondition:
                                key = Transition.RIGHT_ARC + ":" + rel
                                self._write_to_file(key, binary_features, input_file)
                                operation.right_arc(conf, rel)
                                training_seq.append(key)
                                continue

                    # Shift operation as the default
                    key = Transition.SHIFT
                    self._write_to_file(key, binary_features, input_file)
                    operation.shift(conf)
                    training_seq.append(key)

            print(" Number of training examples : " + str(len(depgraphs)))
            print(" Number of valid (projective) examples : " + str(count_proj))
            return training_seq

        def _create_training_examples_arc_eager(self, depgraphs, input_file):
            operation = Transition(self.ARC_EAGER)
            countProj = 0
            training_seq = []

            for depgraph in depgraphs:
                if not self._is_projective(depgraph):
                    continue

                countProj += 1
                conf = MyConfiguration(depgraph)
                while len(conf.buffer) > 0:
                    b0 = conf.buffer[0]
                    features = conf.extract_features()
                    binary_features = self._convert_to_binary_features(features)

                    if len(conf.stack) > 0:
                        s0 = conf.stack[len(conf.stack) - 1]
                        # Left-arc operation
                        rel = self._get_dep_relation(b0, s0, depgraph)
                        if rel is not None:
                            key = Transition.LEFT_ARC + ":" + rel
                            self._write_to_file(key, binary_features, input_file)
                            operation.left_arc(conf, rel)
                            training_seq.append(key)
                            continue

                        # Right-arc operation
                        rel = self._get_dep_relation(s0, b0, depgraph)
                        if rel is not None:
                            key = Transition.RIGHT_ARC + ":" + rel
                            self._write_to_file(key, binary_features, input_file)
                            operation.right_arc(conf, rel)
                            training_seq.append(key)
                            continue

                        # reduce operation
                        flag = False
                        for k in range(s0):
                            if self._get_dep_relation(k, b0, depgraph) is not None:
                                flag = True
                            if self._get_dep_relation(b0, k, depgraph) is not None:
                                flag = True
                        if flag:
                            key = Transition.REDUCE
                            self._write_to_file(key, binary_features, input_file)
                            operation.reduce(conf)
                            training_seq.append(key)
                            continue

                    # Shift operation as the default
                    key = Transition.SHIFT
                    self._write_to_file(key, binary_features, input_file)
                    operation.shift(conf)
                    training_seq.append(key)

            print(" Number of training examples : " + str(len(depgraphs)))
            print(" Number of valid (projective) examples : " + str(countProj))
            return training_seq

        def parse(self, depgraphs, modelFile):
            result = []
            # First load the model
            model = pickle.load(open(modelFile, "rb"))
            operation = Transition(self._algorithm)

            for depgraph in depgraphs:
                conf = MyConfiguration(depgraph)
                while len(conf.buffer) > 0:
                    features = conf.extract_features()
                    col = []
                    row = []
                    data = []
                    for feature in features:
                        if feature in self._dictionary:
                            col.append(self._dictionary[feature])
                            row.append(0)
                            data.append(1.0)
                    np_col = array(sorted(col))  # NB : index must be sorted
                    np_row = array(row)
                    np_data = array(data)

                    x_test = sparse.csr_matrix(
                        (np_data, (np_row, np_col)), shape=(1, len(self._dictionary))
                    )

                    # We will use predict_proba instead of decision_function
                    prob_dict = {}
                    pred_prob = model.predict_proba(x_test)[0]
                    for i in range(len(pred_prob)):
                        prob_dict[i] = pred_prob[i]
                    sorted_Prob = sorted(prob_dict.items(), key=itemgetter(1), reverse=True)

                    # Note that SHIFT is always a valid operation
                    for (y_pred_idx, confidence) in sorted_Prob:
                        # y_pred = model.predict(x_test)[0]
                        # From the prediction match to the operation
                        y_pred = model.classes_[y_pred_idx]

                        if y_pred in self._match_transition:
                            strTransition = self._match_transition[y_pred]
                            baseTransition = strTransition.split(":")[0]

                            if baseTransition == Transition.LEFT_ARC:
                                if (
                                        operation.left_arc(conf, strTransition.split(":")[1])
                                        != -1
                                ):
                                    break
                            elif baseTransition == Transition.RIGHT_ARC:
                                if (
                                        operation.right_arc(conf, strTransition.split(":")[1])
                                        != -1
                                ):
                                    break
                            elif baseTransition == Transition.REDUCE:
                                if operation.reduce(conf) != -1:
                                    break
                            elif baseTransition == Transition.SHIFT:
                                if operation.shift(conf) != -1:
                                    break
                        else:
                            raise ValueError(
                                "The predicted transition is not recognized, expected errors"
                            )

                # Finish with operations build the dependency graph from Conf.arcs
                new_depgraph = deepcopy(depgraph)
                for key in new_depgraph.nodes:
                    node = new_depgraph.nodes[key]
                    node["rel"] = ""
                    # With the default, all the token depend on the Root
                    node["head"] = 0
                for (head, rel, child) in conf.arcs:
                    c_node = new_depgraph.nodes[child]
                    c_node["head"] = head
                    c_node["rel"] = rel
                result.append(new_depgraph)

            return result

    # Evaluate the features comparing performance to the original
    transition_parser = TransitionParser("arc-standard")
    transition_parser.train(dependency_treebank.parsed_sents()[:100], "transition_parser.model")
    parses = transition_parser.parse(dependency_treebank.parsed_sents()[-10:], "transition_parser.model")
    print(len(parses))
    dependency_evaluator = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-10:])
    print(f"The scores of the standard TransitionParser are: {dependency_evaluator.eval()}")
    print()

    my_transition_parser = MyTransitionParser("arc-standard")
    my_transition_parser.train(dependency_treebank.parsed_sents()[:100], "my_transition_parser.model")
    parses = my_transition_parser.parse(dependency_treebank.parsed_sents()[-10:], "my_transition_parser.model")
    print(len(parses))
    dependency_evaluator = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-10:])
    print(f"The scores of MyTransitionParser are: {dependency_evaluator.eval()}")
    print()

    # Replace SVM classifier with an alternative of your choice
    class MyGBCTransitionParser(MyTransitionParser):

        def train(self, depgraphs, modelfile, verbose=True):
            try:
                input_file = tempfile.NamedTemporaryFile(
                    prefix="transition_parse.train", dir=tempfile.gettempdir(), delete=False
                )

                if self._algorithm == self.ARC_STANDARD:
                    self._create_training_examples_arc_std(depgraphs, input_file)
                else:
                    self._create_training_examples_arc_eager(depgraphs, input_file)

                input_file.close()
                # Using the temporary file to train the libsvm classifier
                x_train, y_train = load_svmlight_file(input_file.name)
                model = GradientBoostingClassifier(
                    loss="deviance",
                    learning_rate=0.1,
                    verbose=verbose
                )
                model.fit(x_train, y_train)
                # Save the model to file name (as pickle)
                pickle.dump(model, open(modelfile, "wb"))
            finally:
                remove(input_file.name)

    my_gbc_transition_parser = MyGBCTransitionParser("arc-standard")
    my_gbc_transition_parser.train(dependency_treebank.parsed_sents()[:100], "my_gbc_transition_parser.model")
    parses = my_gbc_transition_parser.parse(dependency_treebank.parsed_sents()[-10:], "my_gbc_transition_parser.model")
    print(len(parses))
    dependency_evaluator = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-10:])
    print(f"The scores of MyGBCTransitionParser are: {dependency_evaluator.eval()}")
    print()
