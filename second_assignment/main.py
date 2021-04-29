from __future__ import absolute_import, annotations

from typing import List, Union, Tuple, Dict

import pandas as pd
import spacy
from sklearn.metrics import classification_report
from spacy import Language
from spacy.tokens import Doc, Span, Token

from conll import get_chunks, read_corpus_conll, evaluate


spacy_nlp: Language = spacy.load("en_core_web_sm")


# 1. Evaluate spaCy NER on CoNLL 2003 dataset (provided)

# spaCy NER labels are different from the one of the CoNLL 2003 dataset, I had to convert some of them and ignore others
print(f"spaCy NER labels: {set(spacy_nlp.get_pipe('ner').labels)}")
print(f"CoNLL 2003 labels: {get_chunks('data/conll2003/test.txt', fs=' ', otag='O')}")
print()

spacy_ner_label_to_conll: Dict[str, str] = {
    "CARDINAL": "",
    "DATE": "",
    "EVENT": "MISC",
    "FAC": "LOC",
    "GPE": "LOC",
    "LANGUAGE": "MISC",
    "LAW": "",
    "LOC": "LOC",
    "MONEY": "",
    "NORP": "MISC",
    "ORDINAL": "",
    "ORG": "ORG",
    "PERCENT": "",
    "PERSON": "PER",
    "PRODUCT": "ORG",
    "QUANTITY": "",
    "TIME": "",
    "WORK_OF_ART": ""
}


# 2. Grouping of Entities

# function to group recognized named entities using `noun_chunks` method of spaCy
def group_named_entities(doc: Union[str, Doc], use_conll_labels: bool = False) -> List[List[str]]:
    if isinstance(doc, str):
        doc: Doc = spacy_nlp(doc)  # since both `ents` and `noun_chunks` are properties of `Doc` object
    elif not isinstance(doc, Doc):
        raise TypeError("You pass a `doc` parameter of a wrong type")

    if not isinstance(use_conll_labels, bool):
        raise TypeError("You pass a `use_conll_labels` parameter of a wrong type")

    noun_chunks: List[Span] = []
    for noun_chunk in doc.noun_chunks:
        noun_chunks.append(noun_chunk)

    chunk: int = 0
    grouped_entities: List[List[str]] = []
    entity_group: List[str] = []
    for ent in doc.ents:
        try:
            if ent in noun_chunks[chunk].ents:
                if use_conll_labels:
                    if spacy_ner_label_to_conll[ent.label_]:
                        entity_group.append(spacy_ner_label_to_conll[ent.label_])
                else:
                    entity_group.append(ent.label_)
            elif entity_group:
                grouped_entities.append(entity_group)
                entity_group: List[str] = []
                chunk += 1
                if ent in noun_chunks[chunk].ents:
                    if use_conll_labels:
                        if spacy_ner_label_to_conll[ent.label_]:
                            entity_group.append(spacy_ner_label_to_conll[ent.label_])
                    else:
                        entity_group.append(ent.label_)
                else:  # some entities may be not within `noun_chunk` spans
                    if use_conll_labels:
                        if spacy_ner_label_to_conll[ent.label_]:
                            grouped_entities.append([spacy_ner_label_to_conll[ent.label_]])
                    else:
                        grouped_entities.append([ent.label_])
            else:  # some entities may be not within `noun_chunk` spans
                if use_conll_labels:
                    if spacy_ner_label_to_conll[ent.label_]:
                        grouped_entities.append([spacy_ner_label_to_conll[ent.label_]])
                else:
                    grouped_entities.append([ent.label_])
        except IndexError:  # some entities may be not within `noun_chunk` spans
            if use_conll_labels:
                if spacy_ner_label_to_conll[ent.label_]:
                    grouped_entities.append([spacy_ner_label_to_conll[ent.label_]])
            else:
                grouped_entities.append([ent.label_])

    if entity_group:
        grouped_entities.append(entity_group)

    return grouped_entities  # the output is a list-of-lists where outer list is the list of groups/chunks and the inner lists are lists of entity labels


# 3. Fix segmentation errors

# function that extends the entity span to cover the full noun-compounds
def extend_entity_span(doc: Union[str, Doc], use_head_compound: bool = False, use_children_compound: bool = False, use_conll_labels: bool = False) -> List[Tuple[str, str]]:
    if isinstance(doc, str):
        doc: Doc = spacy_nlp(doc)  # since `ents` are a property of `Doc` object and with it we have access to all sentence's tokens
    elif not isinstance(doc, Doc):
        raise TypeError("You pass a `doc` parameter of a wrong type")

    if not isinstance(use_head_compound, bool):
        raise TypeError("You pass a `use_head_compound` parameter of a wrong type")

    if not isinstance(use_children_compound, bool):
        raise TypeError("You pass a `use_children_compound` parameter of a wrong type")

    if not isinstance(use_conll_labels, bool):
        raise TypeError("You pass a `use_conll_labels` parameter of a wrong type")

    entities: Dict[int, Tuple[str, str]] = {}
    for ent in doc.ents:
        entity: Dict[int, str] = {}
        for entity_token in ent:
            entity[entity_token.i] = entity_token.text
            token_to_check = entity_token
            if use_head_compound:  # look if the entity tokens are in `compound` dependency relation with other tokens and look for the head from which `compound` relations are originated
                while token_to_check.dep_ == "compound":
                    token_to_check = token_to_check.head
                    entity[token_to_check.i] = token_to_check.text
            if use_children_compound:  # look if the children tokens have a `compound` dependency relation with the entity tokens (or if `use_head_compound` is True with the head from which `compound` relations are originated)
                entity = check_children(token_to_check, entity)

        keys: List[int] = list(entity.keys())
        keys.sort()
        if use_conll_labels:
            if spacy_ner_label_to_conll[ent.label_]:
                entities[keys.pop(0)] = (entity[keys[0]], f"B-{spacy_ner_label_to_conll[ent.label_]}")
        else:
            entities[keys.pop(0)] = (entity[keys[0]], f"B-{ent.label_}")

        for key in keys:
            if use_conll_labels:
                if spacy_ner_label_to_conll[ent.label_]:
                    entities[key] = (entity[key], f"I-{spacy_ner_label_to_conll[ent.label_]}")
            else:
                entities[key] = (entity[key], f"I-{ent.label_}")

    extended_entity_spans: List[Tuple[str, str]] = []
    for doc_token in doc:
        if doc_token.i in entities:
            extended_entity_spans.append(entities[doc_token.i])
        else:
            extended_entity_spans.append((doc_token.text, "O"))

    return extended_entity_spans  # the output is a list-of-tuples where the list has the length of the tokens in the sentence and the tuples contain the token and the iob + entity label


def check_children(token_to_check: Token, entity: Dict[int, str]) -> Dict[int, str]:
    for child in token_to_check.children:
        if child.dep_ == "compound":
            entity[child.i] = child.text
            entity = check_children(child, entity)
    return entity


if __name__ == "__main__":

    test_sentences = read_corpus_conll("data/conll2003/test.txt", fs=" ")
    refs: List[List[Tuple[str, str]]] = [[(text, iob) for text, pos, chunk, iob in sent] for sent in test_sentences]

    spacy_hyps: List[Doc] = []
    for ref in refs:
        sentence: str = ""
        for text, iob in ref:
            sentence = sentence + text + " "
        spacy_hyps.append(spacy_nlp(sentence))

    hyps: List[List[Tuple[str, str]]] = []
    for spacy_hyp in spacy_hyps:
        hyp: List[Tuple[str, str]] = []
        unified_token: List[str, str] = []
        for token in spacy_hyp:
            if not token.whitespace_:
                if not unified_token:
                    unified_token = [token.text, f"{token.ent_iob_}-{spacy_ner_label_to_conll[token.ent_type_]}" if token.ent_type_ and spacy_ner_label_to_conll[token.ent_type_] else "O"]
                else:
                    unified_token[0] = unified_token[0] + token.text
            else:
                if not unified_token:
                    hyp.append((token.text, f"{token.ent_iob_}-{spacy_ner_label_to_conll[token.ent_type_]}" if token.ent_type_ and spacy_ner_label_to_conll[token.ent_type_] else "O"))
                else:
                    unified_token[0] = unified_token[0] + token.text
                    hyp.append(tuple(unified_token))
                    unified_token: List[str, str] = []

        hyps.append(hyp)

    # token-level performance (per class and total)
    token_level_performance = classification_report([token[-1] for sent in refs for token in sent], [token[-1] for sent in hyps for token in sent], digits=3)
    print(f"token-level performances:")
    print(token_level_performance)
    print()

    # chunk-level performance (per class and total)
    chunk_level_performances = evaluate(refs, hyps)
    print("chunk-level performances:")
    print(pd.DataFrame().from_dict(chunk_level_performances, orient="index").round(decimals=3))
    print()

    # test function to group recognized named entities using `noun_chunks` method of spaCy
    print("result of `group_named_entities` on `Apple's Steve Jobs died in 2011 in Palo Alto, California.`:")
    print(group_named_entities("Apple's Steve Jobs died in 2011 in Palo Alto, California.", use_conll_labels=False))
    print()

    # analyze the groups in terms of most frequent combinations (i.e. NER types that go together)
    # frequency analysis of the groups in CoNLL 2003: inner lists are groups, I simply count their frequencies
    frequency_analysis: Dict[Tuple[str], int] = {}
    frequency_analysis_conll_labels: Dict[Tuple[str], int] = {}
    for spacy_hyp in spacy_hyps:
        for entity_list in group_named_entities(spacy_hyp, use_conll_labels=False):
            if tuple(entity_list) in frequency_analysis:
                frequency_analysis[tuple(entity_list)] += 1
            else:
                frequency_analysis[tuple(entity_list)] = 1

        for entity_list in group_named_entities(spacy_hyp, use_conll_labels=True):
            if tuple(entity_list) in frequency_analysis_conll_labels:
                frequency_analysis_conll_labels[tuple(entity_list)] += 1
            else:
                frequency_analysis_conll_labels[tuple(entity_list)] = 1

    print("frequency analysis of the groups in CoNLL 2003:")
    print(frequency_analysis)
    print()

    print("frequency analysis of the groups in CoNLL 2003 using CoNLL 2003 dataset labels:")
    print(frequency_analysis_conll_labels)
    print()

    # test function to extends the entity span to cover the full noun-compounds
    print("result of `extend_entity_span` on `Apple's Steve Jobs died in 2011 in Palo Alto, California.`:")
    print(extend_entity_span("Apple's Steve Jobs died in 2011 in Palo Alto, California.", use_head_compound=True, use_children_compound=True))
    print()

    # evaluate the post-processing on CoNLL 2003 dataset
    hyps_head: List[List[Tuple[str, str]]] = []
    hyps_children: List[List[Tuple[str, str]]] = []
    hyps_head_and_children: List[List[Tuple[str, str]]] = []
    for spacy_hyp in spacy_hyps:
        extended_entity_span_head = extend_entity_span(spacy_hyp, use_head_compound=True, use_conll_labels=True)
        extended_entity_span_children = extend_entity_span(spacy_hyp, use_children_compound=True, use_conll_labels=True)
        extended_entity_span_head_and_children = extend_entity_span(spacy_hyp, use_head_compound=True, use_children_compound=True, use_conll_labels=True)
        hyp_head: List[Tuple[str, str]] = []
        hyp_children: List[Tuple[str, str]] = []
        hyp_head_and_children: List[Tuple[str, str]] = []
        unified_token_head: List[str, str] = []
        unified_token_children: List[str, str] = []
        unified_token_head_and_children: List[str, str] = []
        for token in spacy_hyp:
            if not token.whitespace_:
                if not unified_token_head:
                    unified_token_head = [token.text, extended_entity_span_head[token.i][1]]
                else:
                    unified_token_head[0] = unified_token_head[0] + token.text

                if not unified_token_children:
                    unified_token_children = [token.text, extended_entity_span_children[token.i][1]]
                else:
                    unified_token_children[0] = unified_token_children[0] + token.text

                if not unified_token_head_and_children:
                    unified_token_head_and_children = [token.text, extended_entity_span_head_and_children[token.i][1]]
                else:
                    unified_token_head_and_children[0] = unified_token_head_and_children[0] + token.text

            else:
                if not unified_token_head:
                    hyp_head.append((token.text, extended_entity_span_head[token.i][1]))
                else:
                    unified_token_head[0] = unified_token_head[0] + token.text
                    hyp_head.append(tuple(unified_token_head))
                    unified_token_head: List[str, str] = []

                if not unified_token_children:
                    hyp_children.append((token.text, extended_entity_span_children[token.i][1]))
                else:
                    unified_token_children[0] = unified_token_children[0] + token.text
                    hyp_children.append(tuple(unified_token_children))
                    unified_token_children: List[str, str] = []

                if not unified_token_head_and_children:
                    hyp_head_and_children.append((token.text, extended_entity_span_head_and_children[token.i][1]))
                else:
                    unified_token_head_and_children[0] = unified_token_head_and_children[0] + token.text
                    hyp_head_and_children.append(tuple(unified_token_head_and_children))
                    unified_token_head_and_children: List[str, str] = []

        hyps_head.append(hyp_head)
        hyps_children.append(hyp_children)
        hyps_head_and_children.append(hyp_head_and_children)

    token_level_performance = classification_report([token[-1] for sent in refs for token in sent], [token[-1] for sent in hyps_head for token in sent], digits=3)
    print(f"token-level performances head:")
    print(token_level_performance)
    print()

    chunk_level_performances = evaluate(refs, hyps_head)
    print("chunk-level performances head:")
    print(pd.DataFrame().from_dict(chunk_level_performances, orient="index").round(decimals=3))
    print()

    token_level_performance = classification_report([token[-1] for sent in refs for token in sent], [token[-1] for sent in hyps_children for token in sent], digits=3)
    print(f"token-level performances children:")
    print(token_level_performance)
    print()

    chunk_level_performances = evaluate(refs, hyps_children)
    print("chunk-level performances children:")
    print(pd.DataFrame().from_dict(chunk_level_performances, orient="index").round(decimals=3))
    print()

    token_level_performance = classification_report([token[-1] for sent in refs for token in sent], [token[-1] for sent in hyps_head_and_children for token in sent], digits=3)
    print(f"token-level performances head + children:")
    print(token_level_performance)
    print()

    chunk_level_performances = evaluate(refs, hyps_head_and_children)
    print("chunk-level performances head + children:")
    print(pd.DataFrame().from_dict(chunk_level_performances, orient="index").round(decimals=3))
    print()
