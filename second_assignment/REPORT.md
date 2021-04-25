# Report Second Assignment

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [0. Evaluate spaCy NER on CoNLL 2003 dataset](#0-evaluate-spacy-ner-on-conll-2003-dataset)
  - [token-level performance](#token-level-performance)
  - [chunk-level performance](#chunk-level-performance)
- [1. Grouping of Entities](#1-grouping-of-entities)
  - [function to group recognized named entities using `noun_chunks` method of spaCy](#function-to-group-recognized-named-entities-using-noun_chunks-method-of-spacy)
  - [analyze the groups in terms of most frequent combinations (i.e. NER types that go together)](#analyze-the-groups-in-terms-of-most-frequent-combinations-ie-ner-types-that-go-together)
- [2. Fix segmentation errors](#2-fix-segmentation-errors)
  - [function that extends the entity span to cover the full noun-compounds](#function-that-extends-the-entity-span-to-cover-the-full-noun-compounds)
  - [evaluate the post-processing on CoNLL 2003 data](#evaluate-the-post-processing-on-conll-2003-data)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

The assignment is on the intersection of Named Entity Recognition and Dependency Parsing.

0. Evaluate spaCy NER on [CoNLL 2003 dataset](data/conll2003) (provided):
    * report token-level performance (per class and total)
        * accuracy of correctly recognizing all tokens that belong to named entities (i.e. tag-level accuracy)
    * report chunk-level performance (per class and total)
        * precision, recall, f-measure of correctly recognizing all the named entities in a chunk per class and total

1. Grouping of Entities.
Write a function to group recognized named entities using `noun_chunks` method of [spaCy](https://spacy.io/usage/linguistic-features#noun-chunks).
Analyze the groups in terms of most frequent combinations (i.e. NER types that go together).

2. One of the possible post-processing steps is to fix segmentation errors.
Write a function that extends the entity span to cover the full noun-compounds. Make use of `compound` dependency relation.
Evaluate the post-processing on [CoNLL 2003 dataset](data/conll2003).


## 0. Evaluate spaCy NER on [CoNLL 2003 dataset](data/conll2003)

spaCy NER labels are different from the one of the CoNLL 2003 dataset, I had to convert some of them and ignore others. To do so I used the following dictionary:

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


### token-level performance

The accuracy of correctly recognizing all tokens that belong to named entities (i.e. tag-level accuracy) that I got is:

```python
tag-level accuracy: 0.910063
```


### chunk-level performance

The precision, recall, f-measure of correctly recognizing all the named entities in a chunk per class and total that I got are:

```python
       precision    recall  f1 score  support
LOC     0.747832  0.672062  0.707925     1668
MISC    0.810235  0.541311  0.649018      702
ORG     0.452055  0.278146  0.344391     1661
PER     0.774194  0.608534  0.681440     1617
total   0.691622  0.521778  0.594813     5648
```


## 1. Grouping of Entities

### function to group recognized named entities using `noun_chunks` method of [spaCy](https://spacy.io/usage/linguistic-features#noun-chunks)

The `group_named_entities` function takes as input parameter `doc` (that must be of the one of the following types `str`, `Doc`, otherwise a `TypeError` is raised), and the `use_conll_labels` for deciding which labels to use and returns a list-of-lists where outer list is the list of groups/chunks and the inner lists are lists of entity labels.

```python
def group_named_entities(doc: Union[str, Doc], use_conll_labels: bool = False) -> List[List[str]]:
    if isinstance(doc, str):
        doc: Doc = spacy_nlp(doc)  # Since both `ents` and `noun_chunks` are properties of `Doc` object
    elif not isinstance(doc, Doc):
        raise TypeError("You pass a `doc` parameter of a wrong type")

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
                try:
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
                except IndexError:  # some entities may be not within `noun_chunk` spans
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
I check if the entity is present inside the first noun_chunk and if so I add it to the `entity_group` list and if not I add it directly to the `grouped_entities` list.
Going to the next entity it is the same except for the case in which entity is not present inside the first noun_chunk and `entity_group` is non empty, in this case I add `entity_group` to the `grouped_entities` and I check if the entity is present inside the next noun_chunk.
I repeat this process for all the entities in the doc and in the end I check if `entity_group` is non empty and has to be added to the `grouped_entities` list and then I return the `grouped_entities` list.

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


## 2. Fix segmentation errors

### function that extends the entity span to cover the full noun-compounds

The `extend_entity_span` function takes as input parameter `doc` (that must be of the one of the following types `str`, `Doc`, otherwise a `TypeError` is raised), and the `use_conll_labels` for deciding which labels to use and returns a list-of-tuples where the list has the length of the tokens in the sentence and the tuples contain the token and the IOB + entity label.

```python
def extend_entity_span(doc: Union[str, Doc], use_conll_labels: bool = False) -> List[Tuple[str, str]]:
    if isinstance(doc, str):
        doc: Doc = spacy_nlp(doc)  # Since `ents` are a property of `Doc` object and with it we have access to all sentence's tokens
    elif not isinstance(doc, Doc):
        raise TypeError("You pass a `doc` parameter of a wrong type")

    entities: Dict[int, Tuple[str, str]] = {}
    for ent in doc.ents:
        entity: Dict[int, str] = {}
        for entity_token in ent:
            entity[entity_token.i] = entity_token.text
            for child in entity_token.children:
                if child.dep_ == "compound":  # find the tokens having a `compound` dependency relation with a token of the entity
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
I store its position in the sentence and its text into the `entity` dictionary and then using the token's `.children` attribute I iterate over its children.
If the dependency relation of a token's child is equal to `'compound'` I store its position in the sentence and its text into the `entity` dictionary.
Then sorting the keys of the `entity` dictionary I assign the correct IOB + entity label and I store them in the `entities` dictionary.
I repeat this process for all the entities and in the end I iterate over all the tokens of the document to assign to the missing tokens that are not present in the `entities` dictionary the `O` tag.

The output of the `extend_entity_span` function for the `"Apple's Steve Jobs died in 2011 in Palo Alto, California."` sentence is:

```python
[('Apple', 'B-ORG'), ("'s", 'O'), ('Steve', 'B-PERSON'), ('Jobs', 'I-PERSON'), ('died', 'O'), ('in', 'O'), ('2011', 'B-DATE'), ('in', 'O'), ('Palo', 'B-GPE'), ('Alto', 'I-GPE'), (',', 'O'), ('California', 'B-GPE'), ('.', 'O')]
```


### evaluate the post-processing on [CoNLL 2003 data](data/conll2003)

The accuracy of correctly recognizing all tokens that belong to named entities (i.e. tag-level accuracy) that I got is:

```python
tag-level accuracy: 0.900356
```

The precision, recall, f-measure of correctly recognizing all the named entities in a chunk per class and total that I got are:

```python
       precision    recall  f1 score  support
LOC     0.737236  0.649281  0.690469     1668
MISC    0.804301  0.532764  0.640960      702
ORG     0.449902  0.275738  0.341919     1661
PER     0.646417  0.513296  0.572216     1617
total   0.648017  0.486013  0.555443     5648
```

In conclusion, the use of `compound` dependency relation in this case has a not really good impact, in fact the performances are a bit lower with respect to the first evaluation using only the spaCy pipeline as it is.
