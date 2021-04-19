# Report First Assignment

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Working with Dependency Graphs (Parses)](#working-with-dependency-graphs-parses)
  - [1. extract a path of dependency relations from the ROOT to a token](#1-extract-a-path-of-dependency-relations-from-the-root-to-a-token)
  - [2. extract subtree of dependents given a token](#2-extract-subtree-of-dependents-given-a-token)
  - [3. check if a given list of tokens (segment of a sentence) forms a subtree](#3-check-if-a-given-list-of-tokens-segment-of-a-sentence-forms-a-subtree)
  - [4. identify head of a span, given its tokens](#4-identify-head-of-a-span-given-its-tokens)
  - [5. extract sentence subject, direct object and indirect object spans](#5-extract-sentence-subject-direct-object-and-indirect-object-spans)
- [Training Transition-Based Dependency Parser (Optional & Advanced)](#training-transition-based-dependency-parser-optional--advanced)
  - [Modify NLTK Transition parser ' s Configuration class to use better features.](#modify-nltk-transition-parser--s-configuration-class-to-use-better-features)
  - [Evaluate the features comparing performance to the original](#evaluate-the-features-comparing-performance-to-the-original)
  - [Replace SVM classifier with an alternative of your choice.](#replace-svm-classifier-with-an-alternative-of-your-choice)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

The objective of the assignment is to learn how to work with dependency graphs by defining functions.

Read [spaCy documentation on dependency parser](https://spacy.io/api/dependencyparser) to learn provided methods.

**Working with Dependency Graphs (Parses)**

Define functions to:
1. extract a path of dependency relations from the ROOT to a token
2. extract subtree of dependents given a token
3. check if a given list of tokens (segment of a sentence) forms a subtree
4. identify head of a span, given its tokens
5. extract sentence subject, direct object and indirect object spans

**Training Transition-Based Dependency Parser (Optional & Advanced)**

* Modify [NLTK Transition parser](https://github.com/nltk/nltk/blob/develop/nltk/parse/transitionparser.py) ' s `Configuration` class to use better features.
* Evaluate the features comparing performance to the original
* Replace `SVM` classifier with an alternative of your choice.


## Working with Dependency Graphs (Parses)

In coding the functions, I tried to keep the functions simple and explain each variable using the `Typing` package.

I also added some `print` when executing the main code in order to show the results of the developed functions.

The following functions are tested with the following example sentence and the following spans:

```python
example_sentence: str = "I saw a man with a telescope, he was looking at the Moon."
example_span: Span = spacy_nlp(example_sentence)[2:7]
wrong_span: Span = spacy_nlp(example_sentence)[5:8]
```


### 1. extract a path of dependency relations from the ROOT to a token

The `extract_path_of_dependency_relations` function takes as input parameter `sentence` (that must be string, otherwise a `TypeError` is raised) and returns the list of dependency relations between each token in the sentence from the root to the token.

```python
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
```

To do that, the sentence is first parsed to get a Doc object of spaCy.
A for loop is used to scan all the tokens.
Using the token's `.dep_` attribute I obtained its dependency relation.
Using the token's `.head` attribute I obtained its head.
I go ahead in extracting dependency relations for a token until I reach the root, the token with `token.dep_ == "ROOT"`.
I reverse the list of extracted dependency relations and I add it to the `sentence_dependency_relations` list.
I repeat this process for all the tokens in the sentence and in the end I return the `sentence_dependency_relations` list.

The output of the `extract_path_of_dependency_relations` function for the example sentence is (here is formatted to make it more readable):

```python
[
  ['ROOT', '--ROOT-->', 'saw', '--nsubj-->', 'I'],
  ['ROOT', '--ROOT-->', 'saw'],
  ['ROOT', '--ROOT-->', 'saw', '--dobj-->', 'man', '--det-->', 'a'],
  ['ROOT', '--ROOT-->', 'saw', '--dobj-->', 'man'],
  ['ROOT', '--ROOT-->', 'saw', '--dobj-->', 'man', '--prep-->', 'with'],
  ['ROOT', '--ROOT-->', 'saw', '--dobj-->', 'man', '--prep-->', 'with', '--pobj-->', 'telescope', '--det-->', 'a'],
  ['ROOT', '--ROOT-->', 'saw', '--dobj-->', 'man', '--prep-->', 'with', '--pobj-->', 'telescope'],
  ['ROOT', '--ROOT-->', 'saw', '--ccomp-->', 'looking', '--punct-->', ','],
  ['ROOT', '--ROOT-->', 'saw', '--ccomp-->', 'looking', '--nsubj-->', 'he'],
  ['ROOT', '--ROOT-->', 'saw', '--ccomp-->', 'looking', '--aux-->', 'was'],
  ['ROOT', '--ROOT-->', 'saw', '--ccomp-->', 'looking'],
  ['ROOT', '--ROOT-->', 'saw', '--ccomp-->', 'looking', '--prep-->', 'at'],
  ['ROOT', '--ROOT-->', 'saw', '--ccomp-->', 'looking', '--prep-->', 'at', '--pobj-->', 'Moon', '--det-->', 'the'],
  ['ROOT', '--ROOT-->', 'saw', '--ccomp-->', 'looking', '--prep-->', 'at', '--pobj-->', 'Moon'],
  ['ROOT', '--ROOT-->', 'saw', '--punct-->', '.']
]
```


### 2. extract subtree of dependents given a token

The `extract_dependents_subtree` function takes as input parameter `sentence` (that must be string, otherwise a `TypeError` is raised) and returns the subtree of dependents of each token contained in the sentence.

```python
def extract_dependents_subtree(sentence: str) -> List[List[str]]:
    if not isinstance(sentence, str):
        raise TypeError("You pass a `sentence` parameter of a wrong type")

    spacy_doc: Doc = spacy_nlp(sentence)  # parse the input sentence and get a Doc object of spaCy
    sentence_dependents_subtrees: List[List[str]] = []
    for token in spacy_doc:  # for each token, extract a subtree of its dependents as a list (ordered w.r.t. sentence order)
        token_dependents_subtree: List[str] = [subtree_token.text for subtree_token in token.subtree]
        sentence_dependents_subtrees.append(token_dependents_subtree)

    return sentence_dependents_subtrees
```

To do that, the sentence is first parsed to get a Doc object of spaCy.
A for loop is used to scan all the tokens.
Using the token's `.subtree` attribute I obtained its subtree, it is already in sentence order and with the input word included.
I add it to the `sentence_dependents_subtrees` list.
I repeat this process for all the tokens in the sentence and in the end I return the `sentence_dependents_subtrees` list.

The output of the `extract_dependents_subtree` function for the example sentence is (here is formatted to make it more readable):

```python
[
  ['I'],
  ['I', 'saw', 'a', 'man', 'with', 'a', 'telescope', ',', 'he', 'was', 'looking', 'at', 'the', 'Moon', '.'],
  ['a'],
  ['a', 'man', 'with', 'a', 'telescope'],
  ['with', 'a', 'telescope'],
  ['a'],
  ['a', 'telescope'],
  [','],
  ['he'],
  ['was'],
  [',', 'he', 'was', 'looking', 'at', 'the', 'Moon'],
  ['at', 'the', 'Moon'],
  ['the'],
  ['the', 'Moon'],
  ['.']
]
```


### 3. check if a given list of tokens (segment of a sentence) forms a subtree

The `check_if_tokens_form_a_subtree` function takes as input parameters `sentence` (that must be string, otherwise a `TypeError` is raised) and `tokens` (that must be of the one of the following types `Span`, `List[Token]`, `List[str]`, `str`, otherwise a `TypeError` is raised) and returns `True` or `False` based on the sequence forming a subtree or not.

```python
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
```

The `tokens` parameter type is checked and the if not of type `List[str]` it is converted in that.
To obtain all the possible subtrees of the sentence we can use the `extract_dependents_subtree` function by passing the sentence.
If the `tokens` parameter is present in the `sentence_dependents_subtrees` list returned by the `extract_dependents_subtree` function `True` is returned, if not `False`.

The output of the `check_if_tokens_form_a_subtree` function for the spans presented before and passed to the function in all the possible types are:

```markdown
Does this example span `a man with a telescope` (passing the span object) form a subtree? `True`
Does this example span `'a man with a telescope'` (passing the span as string) form a subtree? `True`
Does this example span `[a, man, with, a, telescope]` (passing the span as tokens objects) form a subtree? `True`
Does this example span `['a', 'man', 'with', 'a', 'telescope']` (passing the span as tokens strings) form a subtree? `True`
Does this wrong span `a telescope,` (passing the span object) form a subtree? `False`
Does this wrong span `'a telescope,'` (passing the span as string) form a subtree? `False`
Does this wrong span `[a, telescope, ,]` (passing the span as tokens objects) form a subtree? `False`
Does this wrong span `['a', 'telescope', ',']` (passing the span as tokens strings) form a subtree? `False`
```


### 4. identify head of a span, given its tokens

The `identify_head_of_a_span` function takes as input parameter `span` (that must be of the one of the following types `Span`, `List[Token]`, `List[str]`, `str`, otherwise a `TypeError` is raised) and returns the root of that span.

```python
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
```

The `span` parameter type is checked and the if not of type `Span` it is converted in that.
Using the span's `.root` attribute I obtained its root and I returned it.

The output of the `identify_head_of_a_span` function for the example span passed to the function in all the possible types are:

```markdown
The head of this example span `a man with a telescope` (passing the span object) is: `man`
The head of this example span `'a man with a telescope'` (passing the span as string): `man`
The head of this example span `[a, man, with, a, telescope]` (passing the span as tokens objects): `man`
The head of this example span `['a', 'man', 'with', 'a', 'telescope']` (passing the span as tokens strings): `man`
```


### 5. extract sentence subject, direct object and indirect object spans

The `extract_nsubj_dobj_iobj` function takes as input parameter `sentence` (that must be string, otherwise a `TypeError` is raised) and returns a dictionary containing as key the possible dependency relations `'nsubj'`, `'dobj'`, `'iobj'` and as value a list containing the tokens of the span that is related to the dependency relation expressed by the key (if a dependency relation is not present its associated list will be empty).

```python
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
                elif child.dep_ == "dative":  # in spaCy "dative" is used instead of "iobj" (that is deprecated)
                    nsubj_dobj_iobj["iobj"].extend([subtree_token.text for subtree_token in child.subtree])

            break

    return nsubj_dobj_iobj
```

To do that, the sentence is first parsed to get a Doc object of spaCy.
A for loop is used to scan all the tokens.
Using the token's `.dep_` attribute I obtained its dependency relation.
If the dependency relation is equal to one of the key `'nsubj'`, `'dobj'`, `'dative'` (used instead of `'iobj'` that is deprecated) using the token's `.subtree` attribute I obtained its subtree, that is converted in a list containing the tokens of the span that is related to the found dependency relation and stored into the `nsubj_dobj_iobj` dictionary.
I repeat this process for all the tokens in the sentence and in the end I return the `nsubj_dobj_iobj` dictionary.

The output of the `extract_nsubj_dobj_iobj` function for the example sentence is (here is formatted to make it more readable):

```python
{
  'nsubj': ['I'],
  'dobj': ['a', 'man', 'with', 'a', 'telescope'],
  'iobj': []
}
```


## Training Transition-Based Dependency Parser (Optional & Advanced)

### Modify NLTK Transition parser ' s Configuration class to use better features.

I created a `MyConfiguration` class extending the original `Configuration` class and I modified the `extract_features` method.
The token is a dictionary like this:

```python
{'address': 2, 'word': 'Vinken', 'lemma': 'Vinken', 'ctag': 'NNP', 'tag': 'NNP', 'feats': '', 'head': 8, 'deps': defaultdict(<class 'list'>, {'': [1, 3, 6, 7]}), 'rel': ''}
```

Now this method uses almost all these attributes as features (except for `address`, `word` and `ctag`) for the first two tokens in the stack and the first two tokens in the buffer:

```python
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
```

The use of `upper()` is to have case-insensitive features.

I also extracted the tag of the third and fourth token both for stack and buffer:

```python
    token = self._tokens[stack_idx2]
    if self._check_informative(token["tag"]):
        result.append("STK_2_POS_" + token["tag"].upper())
```

As was before, I also took the leftmost and rightmost dependency information both for stack and buffer.

Then I created a `MyTransitionParser` class extending the original `TransitionParser` class and I substituted `Configuration` with `MyConfiguration` in the `_create_training_examples_arc_std`, `_create_training_examples_arc_eager` and `parse` methods. 


### Evaluate the features comparing performance to the original

I evaluated the performance with reference to the original parser and I got a slight improvement:

```markdown
The scores of the standard TransitionParser are: (0.7791666666666667, 0.7791666666666667)
The scores of MyTransitionParser are: (0.8166666666666667, 0.8166666666666667)
```


### Replace SVM classifier with an alternative of your choice.

I created a `MyGBCTransitionParser` class extending the `MyTransitionParser` class and I tried to use `GradientBoostingClassifier` instead of `SVC` and the results was pretty good, in particular if compared with the original `TransitionParser`.
I chose the `GradientBoostingClassifier` because it is easy and really fast if compared to the `SVC` classifier. I evaluated the performance and it allows a better optimization and higher scores:

```markdown
The scores of the standard TransitionParser are: (0.7791666666666667, 0.7791666666666667)
The scores of MyTransitionParser are: (0.8166666666666667, 0.8166666666666667)
The scores of MyGBCTransitionParser are: (0.8458333333333333, 0.8458333333333333)
```
