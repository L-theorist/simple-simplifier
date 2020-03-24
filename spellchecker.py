#Using spell checker a la Norvig

import string
alphabet = string.ascii_lowercase

def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    #deletes = [a + b[1:] for a, b in splits if b]
    #transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    #replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    #inserts = [a + c + b     for a, b in splits for c in alphabet]
    #return set(splits + deletes + transposes + replaces + inserts)
    return splits

def separate_words(word, ref):
    """
    Takes a word and applies edits1 to it. If the resulting words are all in ref, the list containing
    them is returned; otherwise a list containing word is returned.
    """
    cand = edits1(word)
    for word1, word2 in cand:
        if word1 in ref and word2 in ref:
            return [word1, word2]
        else:
             continue
    return [word]

def improve_token_list(token_list, ref, spellchecker):
    """
    Not quite an intuitive function. Takes a list of word tokens, if a word is not in ref but spellchecker
    alters it, word is removed from token_list and the result of spellchecker is appended.
    """
    for word in token_list:
        if word not in ref and spellchecker(word, ref) != [word]:
            token_list.remove(word)
            token_list += spellchecker(word, ref)
    return token_list
