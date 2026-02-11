Yep — this structure works well for your experiment, and your reasoning (“good because date and reg are separated / separate chunks”) is exactly the kind of failure mode chunk overlap is meant to fix.

A couple of quick notes + upgrades to make the edge tests even cleaner:

## What’s good about what you wrote

* **You’re anchoring questions to information that is likely split** (e.g., “Regulation (EU) 2021/1133” vs “of 7 July 2021”).
* **Q2 is a classic boundary test** because “Visa Information System (b)” could be in a different chunk from the part that actually names the regulation.

## Small improvements (to reduce ambiguity)

### Q1 (better edge + less guessable)

Your Q1 is fine, but it can be answered even if the model only sees the date once. Make it require *both* the regulation number and the date (more likely to cross a boundary):

**Improved Q1**

```text
Which Regulation number corresponds to the act adopted on 7 July 2021 that establishes conditions for accessing other EU information systems for the purposes of the Visa Information System?
```

This forces: **(EU) 2021/1133 + 7 July 2021 + purpose**.

### Q2 (make it unambiguous and force retrieval)

Right now, “the regulation mentioned in bullet (b)” is okay, but it’s very short and could be partially answered. Force both the ID and what it does:

**Improved Q2**

```text
Under “Visa Information System”, what is the Regulation number listed in point (b), and what is its stated purpose?
```

This forces: **(EU) 2021/1134 + “reforming the Visa Information System.”**

## Even stronger “edge-spanning” question (one that *must* use both a + b)

If you want a single question that’s almost guaranteed to fail without overlap:

**Edge Q3**

```text
Under “Visa Information System”, which Regulation establishes conditions for accessing other EU information systems, and which Regulation reforms the Visa Information System?
```

Answer must include **both**: 2021/1133 and 2021/1134, plus the correct mapping.

If you want, I can also rewrite these as “hard mode” (less keyword-matching, more paraphrase) to stress the retriever rather than the generator.
