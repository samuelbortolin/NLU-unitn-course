# Report Second Assignment

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [1. Evaluate spaCy NER on CoNLL 2003 dataset](#1-evaluate-spacy-ner-on-conll-2003-dataset)
  - [token-level performances](#token-level-performances)
  - [chunk-level performances](#chunk-level-performances)
- [2. Grouping of Entities](#2-grouping-of-entities)
  - [function to group recognized named entities using `noun_chunks` method of spaCy](#function-to-group-recognized-named-entities-using-noun_chunks-method-of-spacy)
  - [analyze the groups in terms of most frequent combinations (i.e. NER types that go together)](#analyze-the-groups-in-terms-of-most-frequent-combinations-ie-ner-types-that-go-together)
- [3. Fix segmentation errors](#3-fix-segmentation-errors)
  - [function that extends the entity span to cover the full noun-compounds](#function-that-extends-the-entity-span-to-cover-the-full-noun-compounds)
  - [evaluate the post-processing on CoNLL 2003 dataset](#evaluate-the-post-processing-on-conll-2003-dataset)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

The assignment is on the intersection of Named Entity Recognition and Dependency Parsing.

1. Evaluate spaCy NER on [CoNLL 2003 dataset](data/conll2003) (provided):
    * report token-level performances (per class and total)
        * accuracy of correctly recognizing all tokens that belong to named entities (i.e. tag-level accuracy)
    * report chunk-level performances (per class and total)
        * precision, recall, f-measure of correctly recognizing all the named entities in a chunk per class and total

2. Grouping of Entities.
Write a function to group recognized named entities using `noun_chunks` method of [spaCy](https://spacy.io/usage/linguistic-features#noun-chunks).
Analyze the groups in terms of most frequent combinations (i.e. NER types that go together).

3. One of the possible post-processing steps is to fix segmentation errors.
Write a function that extends the entity span to cover the full noun-compounds. Make use of `compound` dependency relation.
Evaluate the post-processing on [CoNLL 2003 dataset](data/conll2003).


## 1. Evaluate spaCy NER on [CoNLL 2003 dataset](data/conll2003)

spaCy NER labels are different from the one of the CoNLL 2003 dataset, I had to convert some of them and ignore others. To do this I used the following dictionary:

```python
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
```

Also the tokenization of spaCy is different from the one of the CoNLL 2003 dataset, I had to align them. To do so, after loading the data I reconstruct the sentences and build the previous tokenization using the token's `.whitepace_` attribute:

```python
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
```

In the end to make the hypothesis comparable with references, I unified the entity IOB using the token's `.ent_iob_` attribute and the entity type using the token's `.ent_type_` attribute.


### token-level performances

The per-class and total token-level performances that I got are:

```python
              precision    recall  f1-score   support

       B-LOC      0.759     0.682     0.719      1668
      B-MISC      0.821     0.548     0.658       702
       B-ORG      0.503     0.309     0.383      1661
       B-PER      0.800     0.629     0.704      1617
       I-LOC      0.535     0.560     0.548       257
      I-MISC      0.636     0.324     0.429       216
       I-ORG      0.415     0.515     0.460       835
       I-PER      0.838     0.788     0.812      1156
           O      0.949     0.982     0.965     38554

    accuracy                          0.910     46666
   macro avg      0.695     0.593     0.631     46666
weighted avg      0.903     0.910     0.904     46666
```


### chunk-level performances

The per-class and total chunk-level performances that I got are:

```python
       precision  recall  f1 score  support
LOC        0.748   0.672     0.708     1668
MISC       0.810   0.541     0.649      702
ORG        0.452   0.278     0.344     1661
PER        0.774   0.609     0.681     1617
total      0.692   0.522     0.595     5648
```


## 2. Grouping of Entities

### function to group recognized named entities using `noun_chunks` method of [spaCy](https://spacy.io/usage/linguistic-features#noun-chunks)

The `group_named_entities` function takes as input parameter `doc` (that must be of the one of the following types `str`, `Doc`, otherwise a `TypeError` is raised), and the `use_conll_labels` (that must be of a boolean, otherwise a `TypeError` is raised) for deciding which labels to use and returns a list-of-lists where outer list is the list of groups/chunks and the inner lists are lists of entity labels.

```python
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
```

To do that, the doc is first parsed to get a Doc object of spaCy unless it is not already so.
A for loop is used to get all noun_chunks using the doc's `.noun_chunks` attribute.
A for loop is used to scan all the entities, obtained using doc's `.ents` attribute.
I check if the entity is present inside the first noun_chunk and if this is the case I add it to the `entity_group` list and if not I add it directly to the `grouped_entities` list.
Going to the next entity it is the same except for the case in which entity is not present inside the first noun_chunk and `entity_group` is non-empty, in this case I add `entity_group` to the `grouped_entities` and I check if the entity is present inside the next noun_chunk.
I repeat this process for all the entities in the doc and in the end I check if `entity_group` is non-empty and has to be added to the `grouped_entities` list and then I return the `grouped_entities` list.

The output of the `group_named_entities` function for the `"Apple's Steve Jobs died in 2011 in Palo Alto, California."` sentence is:

```python
[['ORG', 'PERSON'], ['DATE'], ['GPE'], ['GPE']]
```

### analyze the groups in terms of most frequent combinations (i.e. NER types that go together)

I did a frequency analysis of the groups in the [CoNLL 2003 dataset](data/conll2003), in order to find the most frequent combinations (i.e. NER types that go together).
I applied the `group_named_entities` function using the spaCy labels and since inner lists are groups, I simply counted their frequencies and I got (here is formatted and sorted by value to make it more readable):

```python
{
    ('CARDINAL',): 1694,
    ('GPE',): 1311,
    ('PERSON',): 1159,
    ('DATE',): 1035,
    ('ORG',): 936,
    ('NORP',): 347,
    ('MONEY',): 151,
    ('ORDINAL',): 130,
    ('TIME',): 92,
    ('PERCENT',): 82,
    ('EVENT',): 68,
    ('LOC',): 59,
    ('QUANTITY',): 55,
    ('CARDINAL', 'PERSON'): 29,
    ('GPE', 'PERSON'): 25,
    ('FAC',): 25,
    ('PRODUCT',): 23,
    ('NORP', 'PERSON'): 19,
    ('GPE', 'GPE'): 19,
    ('LAW',): 11,
    ('WORK_OF_ART',): 11,
    ('CARDINAL', 'ORG'): 11,
    ('LANGUAGE',): 9,
    ('GPE', 'PRODUCT'): 9,
    ('CARDINAL', 'GPE'): 8,
    ('CARDINAL', 'NORP'): 8,
    ('ORG', 'PERSON'): 7,
    ('GPE', 'ORG'): 7,
    ('PERSON', 'PERSON'): 6,
    ('ORG', 'DATE'): 5,
    ('NORP', 'ORG'): 5,
    ('CARDINAL', 'CARDINAL'): 3,
    ('ORG', 'CARDINAL'): 3,
    ('CARDINAL', 'PERSON', 'CARDINAL'): 3,
    ('ORG', 'NORP'): 3,
    ('PERSON', 'PERSON', 'PERSON'): 2,
    ('CARDINAL', 'ORDINAL'): 2,
    ('DATE', 'ORG'): 2,
    ('ORG', 'ORDINAL'): 2,
    ('GPE', 'CARDINAL'): 2,
    ('DATE', 'NORP'): 2,
    ('GPE', 'DATE'): 2,
    ('ORG', 'GPE'): 2,
    ('PERSON', 'GPE'): 2,
    ('DATE', 'EVENT'): 2,
    ('GPE', 'PERSON', 'CARDINAL'): 1,
    ('ORDINAL', 'DATE'): 1,
    ('ORG', 'GPE', 'ORDINAL'): 1,
    ('ORG', 'QUANTITY'): 1,
    ('ORDINAL', 'PERSON'): 1,
    ('GPE', 'ORDINAL'): 1,
    ('CARDINAL', 'GPE', 'GPE'): 1,
    ('PERCENT', 'CARDINAL'): 1,
    ('PERSON', 'NORP'): 1,
    ('DATE', 'FAC'): 1,
    ('NORP', 'PERSON', 'DATE'): 1,
    ('PRODUCT', 'GPE'): 1,
    ('CARDINAL', 'DATE'): 1,
    ('NORP', 'LOC'): 1,
    ('ORG', 'ORG'): 1,
    ('MONEY', 'MONEY'): 1,
    ('DATE', 'CARDINAL'): 1,
    ('DATE', 'NORP', 'PERSON'): 1,
    ('ORG', 'WORK_OF_ART'): 1,
    ('GPE', 'ORDINAL', 'PERSON'): 1,
    ('PERSON', 'MONEY'): 1,
    ('PERSON', 'GPE', 'CARDINAL'): 1,
    ('ORDINAL', 'ORG'): 1,
    ('ORDINAL', 'EVENT'): 1,
    ('PERSON', 'ORDINAL'): 1,
    ('EVENT', 'ORDINAL'): 1,
    ('GPE', 'NORP'): 1,
    ('GPE', 'FAC'): 1,
    ('PERSON', 'CARDINAL'): 1
}
```

The most attached couples using the spaCy labels are:

```python
('CARDINAL', 'PERSON'): 29
('GPE', 'PERSON'): 25
('NORP', 'PERSON'): 19
('GPE', 'GPE'): 19
```

Instead, using only the [CoNLL 2003 dataset](data/conll2003) labels I got (here is formatted and sorted by value to make it more readable):

```python
{
    ('LOC',): 1415,
    ('PER',): 1201,
    ('ORG',): 988,
    ('MISC',): 441,
    ('LOC', 'PER'): 24,
    ('MISC', 'PER'): 21,
    ('LOC', 'LOC'): 20,
    ('LOC', 'ORG'): 16,
    ('ORG', 'PER'): 6,
    ('PER', 'PER'): 5,
    ('ORG', 'LOC'): 4,
    ('ORG', 'MISC'): 3,
    ('MISC', 'ORG'): 3,
    ('PER', 'LOC'): 3,
    ('PER', 'PER', 'PER'): 2,
    ('PER', 'MISC'): 1,
    ('ORG', 'ORG'): 1,
    ('LOC', 'MISC'): 1,
}
```

The most attached couples using these labels are:

```python
('LOC', 'PER'): 24
('MISC', 'PER'): 21
('LOC', 'LOC'): 20
('LOC', 'ORG'): 16
```


## 3. Fix segmentation errors

### function that extends the entity span to cover the full noun-compounds

The `extend_entity_span` function takes as input parameter `doc` (that must be of the one of the following types `str`, `Doc`, otherwise a `TypeError` is raised), the `use_head_compound` (that must be of a boolean, otherwise a `TypeError` is raised) for using the tokens that are in `compound` dependency relation with the entity tokens, the `use_children_compound` (that must be of a boolean, otherwise a `TypeError` is raised) for using the child tokens having a `compound` dependency relation with the entity tokens, and the `use_conll_labels` (that must be of a boolean, otherwise a `TypeError` is raised) for deciding which labels to use and returns a list-of-tuples where the list has the length of the tokens in the sentence and the tuples contain the token and the IOB + entity label.

```python
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
            if use_head_compound:
                if entity_token.dep_ == "compound":  # find if the entity tokens are in `compound` dependency relation with other tokens
                    entity[entity_token.head.i] = entity_token.head.text
            if use_children_compound:
                for child in entity_token.children:
                    if child.dep_ == "compound":  # find the child tokens having a `compound` dependency relation with the entity tokens
                        entity[child.i] = child.text

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
```

To do that, the doc is first parsed to get a Doc object of spaCy unless it is not already so.
A for loop is used to scan all the entities, obtained using doc's `.ents` attribute.
A for loop is used to scan all the tokens of an entity.
I store its position in the sentence and its text into the `entity` dictionary.
If `use_head_compound` is set to `True` using the token's `.dep_` attribute I get its dependency relation, if it is equal to `'compound'` I store its head position in the sentence and its text into the `entity` dictionary.
If `use_children_compound` is set to `True` using the token's `.children` attribute I iterate over its children. If the dependency relation of a token's child is equal to `'compound'` I store its position in the sentence and its text into the `entity` dictionary.
Then sorting the keys of the `entity` dictionary I assign the correct IOB + entity label and I store them in the `entities` dictionary.
I repeat this process for all the entities and in the end I iterate over all the tokens of the document to assign to the missing tokens that are not present in the `entities` dictionary the `O` tag.

The output of the `extend_entity_span` function for the `"Apple's Steve Jobs died in 2011 in Palo Alto, California."` sentence is:

```python
[('Apple', 'B-ORG'), ("'s", 'O'), ('Steve', 'B-PERSON'), ('Jobs', 'I-PERSON'), ('died', 'O'), ('in', 'O'), ('2011', 'B-DATE'), ('in', 'O'), ('Palo', 'B-GPE'), ('Alto', 'I-GPE'), (',', 'O'), ('California', 'B-GPE'), ('.', 'O')]
```


### evaluate the post-processing on [CoNLL 2003 dataset](data/conll2003)

The per-class and total token-level performances that I got using the compound relations related to the head of the tokens in the entities are:

```python
              precision    recall  f1-score   support

       B-LOC      0.759     0.682     0.719      1668
      B-MISC      0.821     0.548     0.658       702
       B-ORG      0.503     0.309     0.383      1661
       B-PER      0.800     0.629     0.704      1617
       I-LOC      0.344     0.564     0.428       257
      I-MISC      0.418     0.329     0.368       216
       I-ORG      0.360     0.517     0.424       835
       I-PER      0.804     0.792     0.798      1156
           O      0.950     0.972     0.961     38554

    accuracy                          0.902     46666
   macro avg      0.640     0.594     0.605     46666
weighted avg      0.900     0.902     0.899     46666
```

The per-class and total chunk-level performances that I got using the compound relations related to the head of the tokens in the entities are:

```python
       precision  recall  f1 score  support
LOC        0.685   0.627     0.655     1668
MISC       0.728   0.496     0.590      702
ORG        0.361   0.229     0.280     1661
PER        0.764   0.602     0.674     1617
total      0.634   0.487     0.551     5648
```

The per-class and total token-level performances that I got using the compound relations related to the children of the tokens in the entities are:

```python
              precision    recall  f1-score   support

       B-LOC      0.748     0.656     0.699      1668
      B-MISC      0.819     0.543     0.653       702
       B-ORG      0.500     0.305     0.379      1661
       B-PER      0.666     0.519     0.583      1617
       I-LOC      0.432     0.568     0.491       257
      I-MISC      0.614     0.324     0.424       216
       I-ORG      0.409     0.533     0.463       835
       I-PER      0.704     0.797     0.747      1156
           O      0.949     0.976     0.962     38554

    accuracy                          0.900     46666
   macro avg      0.649     0.580     0.600     46666
weighted avg      0.894     0.900     0.895     46666
```

The per-class and total chunk-level performances that I got using the compound relations related to the children of the tokens in the entities are:

```python
       precision  recall  f1 score  support
LOC        0.737   0.649     0.690     1668
MISC       0.804   0.533     0.641      702
ORG        0.450   0.276     0.342     1661
PER        0.646   0.513     0.572     1617
total      0.648   0.486     0.555     5648
```

The per-class and total token-level performances that I got using both the compound relations related to the head and children of the tokens in the entities are:

```python
              precision    recall  f1-score   support

       B-LOC      0.748     0.656     0.699      1668
      B-MISC      0.819     0.543     0.653       702
       B-ORG      0.500     0.305     0.379      1661
       B-PER      0.666     0.519     0.583      1617
       I-LOC      0.315     0.572     0.406       257
      I-MISC      0.418     0.329     0.368       216
       I-ORG      0.362     0.535     0.432       835
       I-PER      0.684     0.800     0.738      1156
           O      0.950     0.967     0.959     38554

    accuracy                          0.894     46666
   macro avg      0.607     0.581     0.580     46666
weighted avg      0.892     0.894     0.890     46666
```

The per-class and total chunk-level performances that I got using both the compound relations related to the head and children of the tokens in the entities are:

```python
       precision  recall  f1 score  support
LOC        0.684   0.612     0.646     1668
MISC       0.730   0.493     0.588      702
ORG        0.368   0.231     0.284     1661
PER        0.639   0.509     0.567     1617
total      0.599   0.456     0.518     5648
```

In conclusion, the use of `compound` dependency relation in this case has a not really positive impact, in fact the performances are a bit lower with respect to the first evaluation using only the spaCy pipeline as it is.
Among these possible approaches using the `compound` dependency relation:
* The better solution in terms of token-level performances seems using only the compound relations related to the head of the tokens in the entities.
* The better solution in terms of chunk-level performances seems using only the compound relations related to the children of the tokens in the entities.
* The worst approach in terms of performances seems using both head and children of the tokens in the entities.
