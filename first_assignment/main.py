from __future__ import absolute_import, annotations

from typing import List, Dict, Union

from nltk.tokenize.treebank import TreebankWordDetokenizer
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
def extract_nsubj_dobj_iobj(sentence: str) -> Dict[str, List[str]]:
    if not isinstance(sentence, str):
        raise TypeError("You pass a `sentence` parameter of a wrong type")

    spacy_doc: Doc = spacy_nlp(sentence)  # parse the input sentence and get a Doc object of spaCy
    nsubj_dobj_iobj: Dict[str, List[str]] = dict({"nsubj": list(), "dobj": list(), "iobj": list()})  # output is dict of lists of words that form a span for subject, direct object, and indirect object (if present, otherwise empty)
    for token in spacy_doc:
        if token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ == "nsubj":
                    nsubj_dobj_iobj["nsubj"].extend([subtree_token.text for subtree_token in child.subtree])
                elif child.dep_ == "dobj":
                    nsubj_dobj_iobj["dobj"].extend([subtree_token.text for subtree_token in child.subtree])
                elif child.dep_ == "iobj":
                    nsubj_dobj_iobj["iobj"].extend([subtree_token.text for subtree_token in child.subtree])

            break

    return nsubj_dobj_iobj


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
    print(extract_nsubj_dobj_iobj(example_sentence))
    print()
