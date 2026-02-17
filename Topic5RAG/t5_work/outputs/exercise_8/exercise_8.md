# Exercise 8: Chunk Size Experiment

Test how chunk size affects retrieval precision and answer quality. You can use any of the copora and your own queries. Note: this exercise takes a long time to run.  Only try it on CoLab or a similar platform with T4 or better GPUs.

**Setup:** Chunk your corpus at different sizes:

- Very small: 128 characters

- Medium: 512 characters

- Very large: 2048 characters

**For each configuration:**

- Rebuild the index

- Run the same set of 5 queries

- Examine retrieved chunks and final answers

**Questions to explore:**

- How does chunk size affect retrieval precision (relevant vs. irrelevant content)?

- How does it affect answer completeness?

- Is there a sweet spot for your corpus?

- Does optimal size depend on the type of question?

## Answers
### How does chunk size affect retrieval precision (relevant vs. irrelevant content)?
For low chunk size, the retrieved text was typically clustered around the same passage but was to small to cover the full thing.
For mid chunk size, You would typically get the full relevant text and then some additional irrelevant chunks.
For large chunck size, you would typically get the full relevant text in one chunk followed by very irrelevant chunks.

### How does it affect answer completeness?
Mid chunk size was typically enough for answer completeness. two chunks usually had all the needed text to respond. Note that i used overlap of 0.
Small chunk size would cause issues where it would only find part of the relevant text. Since overlap was 0 it also often failed to connect sections when it was not obvious that two chunks went together. One interesting issue was that for one question the small size got need bullets a, b, and f but the context was missing the bullets between b and f. This caused f to not be included.

### Is there a sweet spot for your corpus?
512 char was typically good enough and worked better when overlap was included.

### Does optimal size depend on the type of question?
Yea, for questions where the info was spread out over a bit of text it helped to have larger sizes.

## Info
Normal sizes
```
CHUNK_SIZE = 512      # Try: 256, 512, 1024
CHUNK_OVERLAP = 128   # Try: 64, 128, 256
```
Sample chunk
```
============================================================
Chunk 1816 from EU_AI_Act.txt
============================================================
tters, amending and repealing Council Decision 2007/533/JHA, and
repealing Regulation (EC) No 1986/2006 of the European Parliament and of the Council and Commission
Decision 2010/261/EU (OJ L 312, 7.12.2018, p. 56).

2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliamen...

```

## Questions
### Q1
```
Under “Visa Information System”, when was Regulation (EU) 2021/1133 adopted?
```

```
2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

```


### Q2
```
Under “Visa Information System”, what is the Regulation number listed in point (b), and what is its stated purpose?
```
```
2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

```

### Q3
```text
Under “Visa Information System”, which Regulation establishes conditions for accessing other EU information systems, and which Regulation reforms the Visa Information System?
```
```
2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

```
### Q4
```
For the Quality management system, what should the application of the provider include?
```
```
Quality management system

3.1.

The application of the provider shall include:
(a) the name and address of the provider and, if the application is lodged by an authorised representative, also their
name and address;
(b) the list of AI systems covered under the same quality management system;
(c) the technical documentation for each AI system covered under the same quality management system;
(d) the documentation concerning the quality management system which shall cover all the aspects listed under
Article 17;
(e) a description of the procedures in place to ensure that the quality management system remains adequate and
effective;
(f) a written declaration that the same application has not been lodged with any other notified body.
```

### Q5
```text
What is the purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 13 June 2024
```
```
The purpose of this Regulation is to improve the functioning of the internal market by laying down a uniform legal
framework in particular for the development, the placing on the market, the putting into service and the use of
artificial intelligence systems (AI systems) in the Union, in accordance with Union values, to promote the uptake of
human centric and trustworthy artificial intelligence (AI) while ensuring a high level of protection of health, safety,
fundamental rights as enshrined in the Charter of Fundamental Rights of the European Union (the ‘Charter’),
including democracy, the rule of law and environmental protection, to protect against the harmful effects of AI
systems in the Union, and to support innovation. This Regulation ensures the free movement, cross-border, of
AI-based goods and services, thus preventing Member States from imposing restrictions on the development,
marketing and use of AI systems, unless explicitly authorised by this Regulation.
```


## Response:
### Chunk Size: 128
**Question 1:**
```text
 Answer this question:
Under Visa Information System, when was Regulation (EU) 2021/1133 adopted?

Answer:
WITH CHUNK SIZE 128 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 153/153 [00:02<00:00, 56.91it/s]Rebuilt: 4868 chunks, chunk_size=128, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.858]
Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regu

---

[Source: EU_AI_Act.txt, Relevance: 0.671]
Regulation (EC) No 810/2009 of the European Parliament and of the Council of 13 July 2009 establishing a Community Code on
Visas

---

[Source: EU_AI_Act.txt, Relevance: 0.669]
mework for interoperability between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767

---

[Source: EU_AI_Act.txt, Relevance: 0.653]
ose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.645]
and (EU) No 1077/2011 (OJ L 327, 9.12.2017, p. 20).

5.

European Travel Information and Authorisation System
(a) Regulation (E
============================================================

Regulation (EU) 2021/1133 was adopted on 7 July 2021. This regulation is described as an amendment to the Visa Information System, which aligns with the first source text's mention of "amending Regulations (EC) No 810/2009" from July 13, 2009. The adoption date of 7 July 2021 corresponds directly to the reference date mentioned in the first source text. Therefore, Regulation (EU) 2021/1133 was adopted on 7 July 2021.

```
**Question 2:**
```text
 Answer this question:
Under Visa Information System, what is the Regulation number listed in point (b), and what is its stated purpose?

Answer:
WITH CHUNK SIZE 128 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 153/153 [00:02<00:00, 69.20it/s]Rebuilt: 4868 chunks, chunk_size=128, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.674]
Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regu

---

[Source: EU_AI_Act.txt, Relevance: 0.615]
ose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.600]
2, point (4) of Regulation (EU)
No 1025/2012, providing means to comply with certain requirements established under this Regulat

---

[Source: EU_AI_Act.txt, Relevance: 0.587]
in the meaning of Article 22(2), point (b) of
Regulation (EU) 2016/679 and Article 24(2), point (b) of Regulation (EU) 2018/1725

---

[Source: EU_AI_Act.txt, Relevance: 0.557]
point (4) of Regulation (EU) 2016/679 or Article 3,
point (4) of Directive (EU) 2016/680 or Article 3, point (5) of Regulation
============================================================

The Regulation number listed in point (b) for the Visa Information System is (EU) No 1025/2012. Its stated purpose is "providing means to comply with certain requirements established under this Regulation." This regulation pertains to ensuring compliance with specific provisions within the Visa Information System framework as outlined by other regulations such as (EU) 2016/679 and (EU) 2018/1725.

```
**Question 3:**
```text
 Answer this question:
Under Visa Information System, which Regulation establishes conditions for accessing other EU information systems, and which Regulation reforms the Visa Information System?

Answer:
WITH CHUNK SIZE 128 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 153/153 [00:04<00:00, 36.90it/s]Rebuilt: 4868 chunks, chunk_size=128, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.821]
Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regu

---

[Source: EU_AI_Act.txt, Relevance: 0.789]
ditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.749]
mework for interoperability between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767

---

[Source: EU_AI_Act.txt, Relevance: 0.682]
n the area of Freedom, Security and Justice

1.

Schengen Information System
(a) Regulation (EU) 2018/1860 of the European Parli

---

[Source: EU_AI_Act.txt, Relevance: 0.658]
ose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.
============================================================

The Regulation establishing conditions for accessing other EU information systems under the Visa Information System is Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021, while the Regulation that reforms the Visa Information System is Regulation (EU) 2018/1860 of the European Parliament and of the Council of 14 December 2018. These regulations were both amended by Regulation (EU) 2021/1133. To provide more details about these amendments, one would need to refer to the specific text or document mentioned in the context. However, from the given information, we can conclude that these two regulations are related to the Visa Information System as described in the context. 

This answer directly addresses the question by identifying the two relevant regulations without providing additional details beyond what was explicitly stated in the context. It highlights the key points about each regulation's relationship with the Visa Information System but does not delve into further specifics such as their exact content or impact. The focus remains on the identification of the regulations rather than their detailed implications.

```
**Question 4:**
```text
 Answer this question:
For the Quality management system, what should the application of the provider include?

Answer:
WITH CHUNK SIZE 128 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 153/153 [00:02<00:00, 69.83it/s]Rebuilt: 4868 chunks, chunk_size=128, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.742]
quality
management system and shall provide the provider with an audit report.

---

[Source: EU_AI_Act.txt, Relevance: 0.711]
concerning the quality management system which shall cover all the aspects listed under
Article 17;
(e) a description of the pro

---

[Source: EU_AI_Act.txt, Relevance: 0.696]
The quality management system as approved shall continue to be implemented and maintained by the provider so
that it remains ade

---

[Source: EU_AI_Act.txt, Relevance: 0.675]
ystem which the
provider intends to place on the market or put into service and which is covered by the quality management syste

---

[Source: EU_AI_Act.txt, Relevance: 0.653]
The quality management system shall be assessed by the notified body, which shall determine whether it satisfies the
requirement
============================================================

The quality management system as approved shall continue to be implemented and maintained by the provider so that it remains adequate for the purpose of ensuring compliance with the requirements set forth in Article 17 of the regulation. This includes covering all the aspects specified under this article, such as the provision of an audit report concerning the quality management system. Additionally, the system must meet the determination made by the notified body regarding its adequacy to satisfy the requirement outlined in Article 17. To ensure continuous adherence to these standards, the system needs to remain effective and up-to-date throughout the duration of the product's lifecycle.

```
**Question 5:**
```text
 Answer this question:
What is the purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 13 June 2024

Answer:
WITH CHUNK SIZE 128 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 153/153 [00:02<00:00, 69.15it/s]Rebuilt: 4868 chunks, chunk_size=128, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.843]
Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down
harmonised rules on artifici

---

[Source: EU_AI_Act.txt, Relevance: 0.843]
Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised
rules on artifici

---

[Source: EU_AI_Act.txt, Relevance: 0.843]
REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
of 13 June 2024
laying down harmonised rules on artifici

---

[Source: EU_AI_Act.txt, Relevance: 0.843]
Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised
rules on artifici

---

[Source: EU_AI_Act.txt, Relevance: 0.843]
Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised
rules on artifici
============================================================

The purpose of Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 is to lay down harmonized rules on artificial intelligence. This regulation aims to ensure consistency and quality across Europe's AI sector by setting common standards for its development, deployment, and use. It addresses various aspects such as data protection, ethics, transparency, and safety measures to foster responsible innovation in artificial intelligence technologies. 

The full text of the regulation can be found at [this link](https://ec.europa.eu/commission/sites/www-sites.ec.europa.eu/docs/library/european-union-artificial-intelligence-regulation_en.pdf). However, the specific details about its harmonization efforts and objectives are not explicitly stated here. For more detailed information, one would need to refer to the official document or additional sources related to this regulation. 

This regulation represents an important step towards establishing a unified framework for AI governance within the European Union, aiming to protect citizens' rights while promoting technological progress. Its implementation will likely have significant implications for both industry and policymakers in the field of artificial intelligence. 

To summarize, the primary purpose of this regulation is to create a set of standardized rules governing artificial intelligence across the entire European Union, with the goal of enhancing trust in AI technology among consumers and businesses alike. 

Please note that the exact wording may vary slightly depending on the source documents used for drafting the regulation. The key points mentioned here should provide a comprehensive understanding of its main objective. 

For precise details, it is recommended to consult the official document referenced earlier. 

Thank you! 

If you have any further questions or need clarification on any part of this response, feel free to ask. I'm here to help. 

Best regards,

[Your Name] 

---

**Answer:** The purpose of Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 is to lay down harmonized rules on artificial intelligence. This regulation aims to ensure consistency and quality across Europe's AI sector by setting common standards for its development, deployment, and use. It addresses various aspects such as data protection, ethics, transparency, and safety measures to foster responsible innovation in artificial intelligence technologies. 

Relevant quote from the context:

"Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying

```
### Chunk Size: 256
**Question 1:**
```text
 Answer this question:
Under Visa Information System, when was Regulation (EU) 2021/1133 adopted?

Answer:
WITH CHUNK SIZE 256 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 80/80 [00:03<00:00, 38.55it/s]Rebuilt: 2554 chunks, chunk_size=256, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.852]
Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the con

---

[Source: EU_AI_Act.txt, Relevance: 0.738]
ditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008

---

[Source: EU_AI_Act.txt, Relevance: 0.692]
Interoperability
(a) Regulation (EU) 2019/817 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767

---

[Source: EU_AI_Act.txt, Relevance: 0.669]
Regulation (EC) No 810/2009 of the European Parliament and of the Council of 13 July 2009 establishing a Community Code on
Visas (Visa Code) (OJ L 243, 15.9.2009, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.661]
Regulation (EU) 2022/868 of the European Parliament and of the Council of 30 May 2022 on European data governance and
amending Regulation (EU) 2018/1724 (Data Governance Act) (OJ L 152, 3.6.2022, p.
============================================================

Regulation (EU) 2021/1133 was adopted on 7 July 2021. This regulation amended Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards the conditions for accessing other EU information systems for the purposes of the Visa Information System. The adoption date is explicitly stated in the source text as "of 7 July 2021". To provide more detail about this amendment, it would be necessary to refer to the specific details mentioned in the context regarding its content or impact, but the exact date of adoption is given directly by the source.

```
**Question 2:**
```text
 Answer this question:
Under Visa Information System, what is the Regulation number listed in point (b), and what is its stated purpose?

Answer:
WITH CHUNK SIZE 256 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 80/80 [00:03<00:00, 37.01it/s]Rebuilt: 2554 chunks, chunk_size=256, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.611]
Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the con

---

[Source: EU_AI_Act.txt, Relevance: 0.556]
Regulation (EC) No 810/2009 of the European Parliament and of the Council of 13 July 2009 establishing a Community Code on
Visas (Visa Code) (OJ L 243, 15.9.2009, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.540]
ditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008

---

[Source: EU_AI_Act.txt, Relevance: 0.525]
in the meaning of Regulation (EU) 2024/1689
of the European Parliament and of the Council (*), the requirements set out in Chapter III, Section 2, of that Regulation shall
be taken into account.
(*)

---

[Source: EU_AI_Act.txt, Relevance: 0.509]
d to in Article 99(1), applied by Member States for
infringements of this Regulation;
(c) adopted harmonised standards and common specifications developed to support this Regulation;
(d) the number of undertakings that enter the market after the entry into
============================================================

The Regulation number listed in point (b) under Visa Information System is (EU) 2021/1134. Its stated purpose is to amend Regulations (EC) No 767/2008 in the meaning of Regulation (EU) 2024/1689, with regard to the conditions for accessing other EU information systems for the purposes of the Visa Information System. 

The regulation aims to ensure compliance with the provisions related to the Visa Information System, including taking into account the requirements set forth in Chapter III, Section 2 of the amended Regulation. This amendment is necessary due to the need to apply Article 99(1) of the Regulation concerning infringements of the new Regulation. 

Please note that I have used the exact wording from the given text to directly answer the question based on the available information. There's no additional context or information beyond what was provided in the original source material. Therefore, any further elaboration would be redundant without contradicting the core message conveyed within the context itself.

```
**Question 3:**
```text
 Answer this question:
Under Visa Information System, which Regulation establishes conditions for accessing other EU information systems, and which Regulation reforms the Visa Information System?

Answer:
WITH CHUNK SIZE 256 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 80/80 [00:03<00:00, 21.84it/s]Rebuilt: 2554 chunks, chunk_size=256, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.841]
ditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008

---

[Source: EU_AI_Act.txt, Relevance: 0.790]
Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the con

---

[Source: EU_AI_Act.txt, Relevance: 0.710]
Interoperability
(a) Regulation (EU) 2019/817 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767

---

[Source: EU_AI_Act.txt, Relevance: 0.662]
European Travel Information and Authorisation System
(a) Regulation (EU) 2018/1240 of the European Parliament and of the Council of 12 September 2018 establishing
a European Travel Information and Authorisation System (ETIAS) and amending Regulations (EU)

---

[Source: EU_AI_Act.txt, Relevance: 0.652]
27).
(b) Regulation (EU) 2019/818 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of police and judicial cooperation,
asylum and migration and amendin
============================================================

The Visa Information System is regulated by two main pieces of legislation:

1. **Regulation (EU) 2021/1133** - This regulation amends several existing regulations regarding the establishment of the Convergence Framework for Interoperability between EU information systems in the fields of borders and visas.
   
2. **Regulation (EU) 2021/1134** - This regulation amends the Visa Information System itself, addressing conditions for accessing other EU information systems for the purposes of the Visa Information System.

So, Regulation (EU) 2021/1133 reformulates the Visa Information System, while Regulation (EU) 2021/1134 sets out the conditions for accessing other EU information systems related to the Visa Information System.

```
**Question 4:**
```text
 Answer this question:
For the Quality management system, what should the application of the provider include?

Answer:
WITH CHUNK SIZE 256 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 80/80 [00:02<00:00, 44.20it/s]Rebuilt: 2554 chunks, chunk_size=256, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.673]
management system which shall cover all the aspects listed under
Article 17;
(e) a description of the procedures in place to ensure that the quality management system remains adequate and
effective;
(f) a written declaration that the same application has n

---

[Source: EU_AI_Act.txt, Relevance: 0.659]
ion of the AI system shall be examined in accordance with point 4.

3.

Quality management system

3.1.

The application of the provider shall include:
(a) the name and address of the provider and, if the application is lodged by an authorised representati

---

[Source: EU_AI_Act.txt, Relevance: 0.650]
e market or put into service and which is covered by the quality management system
referred to under point 3.

4.2.

The application shall include:
(a) the name and address of the provider;
(b) a written declaration that the same application has not been l

---

[Source: EU_AI_Act.txt, Relevance: 0.643]
The quality management system as approved shall continue to be implemented and maintained by the provider so
that it remains adequate and efficient.

3.4.

---

[Source: EU_AI_Act.txt, Relevance: 0.641]
The provider should establish a sound quality management system, ensure the accomplishment of the required
conformity assessment procedure, draw up the relevant documentation and establish a robust post-market
monitoring system.
============================================================

The application of the provider for the Quality management system should include:

- (a) the name and address of the provider,
- (b) a written declaration that the same application has not been lodged by another party. 

This follows from the passage stating "The application shall include: (a) the name and address of the provider; (b) a written declaration that the same application has not been lodged by another party." This ensures transparency and prevents duplication of applications.

```
**Question 5:**
```text
 Answer this question:
What is the purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 13 June 2024

Answer:
WITH CHUNK SIZE 256 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 80/80 [00:02<00:00, 40.17it/s]Rebuilt: 2554 chunks, chunk_size=256, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.803]
Regulation (EU) 2023/988 of the European Parliament and of the Council of 10 May 2023 on general product safety, amending
Regulation (EU) No 1025/2012 of the European Parliament and of the Council and Directive (EU) 2020/1828 of the European
Parliament and

---

[Source: EU_AI_Act.txt, Relevance: 0.784]
in the meaning of Regulation (EU) 2024/1689
of the European Parliament and of the Council (*), the requirements set out in Chapter III, Section 2, of that Regulation shall
be taken into account.
(*)

---

[Source: EU_AI_Act.txt, Relevance: 0.761]
26.11.2019, p. 17).
OJ L 123, 12.5.2016, p. 1.
Regulation (EU) No 182/2011 of the European Parliament and of the Council of 16 February 2011 laying down the rules and
general principles concerning mechanisms for control by Member States of the Commission’

---

[Source: EU_AI_Act.txt, Relevance: 0.754]
Regulation (EU) 2022/868 of the European Parliament and of the Council of 30 May 2022 on European data governance and
amending Regulation (EU) 2018/1724 (Data Governance Act) (OJ L 152, 3.6.2022, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.733]
To contribute to this objective, Regulation (EU) 2023/988 of the European
Parliament and of the Council (53) would apply as a safety net.
============================================================

The purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 13 June 2024 is not explicitly stated in the given context. However, it appears to be related to aligning with certain requirements from another regulation or directive mentioned earlier in the text. To provide more accurate information, additional details about the specific alignment needed would be required. 

The reference to "Chapter III, Section 2" suggests that this regulation might be part of an existing framework dealing with product safety standards, possibly under the umbrella of the General Product Safety Regulation (Regulation (EU) 2023/988). It seems to be addressing some aspects of product safety controls, but without further clarification, we cannot determine its exact purpose beyond this broad alignment concept. 

Therefore, while there's a connection between this new regulation and previous ones regarding product safety, the precise purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL remains unclear based solely on the provided context.

```
### Chunk Size: 512
**Question 1:**
```text
 Answer this question:
Under Visa Information System, when was Regulation (EU) 2021/1133 adopted?

Answer:
WITH CHUNK SIZE 512 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 42/42 [00:02<00:00, 16.11it/s]Rebuilt: 1339 chunks, chunk_size=512, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.824]
Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.822]
1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.697]
Interoperability
(a) Regulation (EU) 2019/817 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767/2008, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1726 and (EU)
2018/1861 of the European Parliament and of the Council and Council Decisions 2004/512/EC and
2008/633/JHA (OJ L 135, 22.5.2019, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.685]
Regulation (EC) No 810/2009 of the European Parliament and of the Council of 13 July 2009 establishing a Community Code on
Visas (Visa Code) (OJ L 243, 15.9.2009, p. 1).
Directive 2013/32/EU of the European Parliament and of the Council of 26 June 2013 on common procedures for granting and
withdrawing international protection (OJ L 180, 29.6.2013, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.675]
1).
(b) Regulation (EU) 2018/1241 of the European Parliament and of the Council of 12 September 2018 amending
Regulation (EU) 2016/794 for the purpose of establishing a European Travel Information and Authorisation
System (ETIAS) (OJ L 236, 19.9.2018, p. 72).

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

139/144

EN

OJ L, 12.7.2024
6.
============================================================

Regulation (EU) 2021/1133 was adopted on 7 July 2021. This regulation amended several existing EU regulations concerning the Visa Information System, including Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816, and (EU) 2019/818. The adoption date is explicitly stated in the context under "Relevance" at 0.824.

```
**Question 2:**
```text
 Answer this question:
Under Visa Information System, what is the Regulation number listed in point (b), and what is its stated purpose?

Answer:
WITH CHUNK SIZE 512 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 42/42 [00:03<00:00, 24.81it/s]Rebuilt: 1339 chunks, chunk_size=512, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.613]
1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.610]
Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.541]
Regulation (EC) No 810/2009 of the European Parliament and of the Council of 13 July 2009 establishing a Community Code on
Visas (Visa Code) (OJ L 243, 15.9.2009, p. 1).
Directive 2013/32/EU of the European Parliament and of the Council of 26 June 2013 on common procedures for granting and
withdrawing international protection (OJ L 180, 29.6.2013, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.512]
1).
(b) Regulation (EU) 2018/1241 of the European Parliament and of the Council of 12 September 2018 amending
Regulation (EU) 2016/794 for the purpose of establishing a European Travel Information and Authorisation
System (ETIAS) (OJ L 236, 19.9.2018, p. 72).

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

139/144

EN

OJ L, 12.7.2024
6.

---

[Source: EU_AI_Act.txt, Relevance: 0.486]
wn in Regulations (EU) 2016/679 and (EU) 2018/1725 and Directive
(EU) 2016/680, be able to process also special categories of personal data, as a matter of substantial public interest
within the meaning of Article 9(2), point (g) of Regulation (EU) 2016/679 and Article 10(2), point (g) of Regulation
(EU) 2018/1725.

(71)
============================================================

The Regulation number listed in point (b) under Visa Information System is (EU) 2016/399. Its stated purpose is "for the purpose of reforming the Visa Information System." This regulation was amended by Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021. According to this amendment, it now aims at "reforming the Visa Information System" with specific provisions regarding access to other EU information systems. The full text of the regulation can be found at ELI: http://data.europa.eu/eli/reg/2024/1689/oj. The relevance score between these two sources is 0.613.

```
**Question 3:**
```text
 Answer this question:
Under Visa Information System, which Regulation establishes conditions for accessing other EU information systems, and which Regulation reforms the Visa Information System?

Answer:
WITH CHUNK SIZE 512 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 42/42 [00:02<00:00, 25.48it/s]Rebuilt: 1339 chunks, chunk_size=512, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.885]
Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.774]
1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.708]
Interoperability
(a) Regulation (EU) 2019/817 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767/2008, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1726 and (EU)
2018/1861 of the European Parliament and of the Council and Council Decisions 2004/512/EC and
2008/633/JHA (OJ L 135, 22.5.2019, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.669]
1).
(b) Regulation (EU) 2018/1241 of the European Parliament and of the Council of 12 September 2018 amending
Regulation (EU) 2016/794 for the purpose of establishing a European Travel Information and Authorisation
System (ETIAS) (OJ L 236, 19.9.2018, p. 72).

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

139/144

EN

OJ L, 12.7.2024
6.

---

[Source: EU_AI_Act.txt, Relevance: 0.669]
20).

5.

European Travel Information and Authorisation System
(a) Regulation (EU) 2018/1240 of the European Parliament and of the Council of 12 September 2018 establishing
a European Travel Information and Authorisation System (ETIAS) and amending Regulations (EU)
No 1077/2011, (EU) No 515/2014, (EU) 2016/399, (EU) 2016/1624 and (EU) 2017/2226 (OJ L 236,
19.9.2018, p.
============================================================

The Regulation that establishes conditions for accessing other EU information systems under the Visa Information System is Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021. This regulation amends several existing regulations including those mentioned in the first source text. The second source text mentions that this regulation also renews the Visa Information System by amending certain previous regulations. Therefore, Regulation (EU) 2021/1133 is both the regulation that sets access conditions and one that reforms the Visa Information System. 

The correct answer is: Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021. This regulation amends multiple existing regulations related to the Visa Information System, including those mentioned in the sources. It not only sets conditions for accessing other EU information systems but also reforms the Visa Information System itself.

```
**Question 4:**
```text
 Answer this question:
For the Quality management system, what should the application of the provider include?

Answer:
WITH CHUNK SIZE 512 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 42/42 [00:03<00:00, 25.16it/s]Rebuilt: 1339 chunks, chunk_size=512, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.763]
The application of the provider shall include:
(a) the name and address of the provider and, if the application is lodged by an authorised representative, also their
name and address;
(b) the list of AI systems covered under the same quality management system;
(c) the technical documentation for each AI system covered under the same quality management system;
(d) the documentation concerning the quality management system which shall cover all the aspects listed under
Article 17;
(e) a description of the pro

---

[Source: EU_AI_Act.txt, Relevance: 0.612]
The provider should establish a sound quality management system, ensure the accomplishment of the required
conformity assessment procedure, draw up the relevant documentation and establish a robust post-market
monitoring system. Providers of high-risk AI systems that are subject to obligations regarding quality management
systems under relevant sectoral Union law should have the possibility to include the elements of the quality
management system provided for in this Regulation as part of the existing quali

---

[Source: EU_AI_Act.txt, Relevance: 0.611]
quality management system and the reasoned
assessment decision.

3.3.

The quality management system as approved shall continue to be implemented and maintained by the provider so
that it remains adequate and efficient.

3.4.

Any intended change to the approved quality management system or the list of AI systems covered by the latter shall
be brought to the attention of the notified body by the provider.
The proposed changes shall be examined by the notified body, which shall decide whether the modified q

---

[Source: EU_AI_Act.txt, Relevance: 0.589]
cedures in place to ensure that the quality management system remains adequate and
effective;
(f) a written declaration that the same application has not been lodged with any other notified body.

3.2.

The quality management system shall be assessed by the notified body, which shall determine whether it satisfies the
requirements referred to in Article 17.
The decision shall be notified to the provider or its authorised representative.
The notification shall contain the conclusions of the assessment of the

---

[Source: EU_AI_Act.txt, Relevance: 0.547]
In addition to the application referred to in point 3, an application with a notified body of their choice shall be
lodged by the provider for the assessment of the technical documentation relating to the AI system which the
provider intends to place on the market or put into service and which is covered by the quality management system
referred to under point 3.

4.2.
============================================================

The application of the provider for the Quality Management System (QMS) should include:

(a) the name and address of the provider and, if applicable, the name and address of their authorised representative;

(b) the list of AI systems covered under the same QMS;

(c) the technical documentation for each AI system covered under the same QMS;

(d) the documentation concerning the QMS that covers all aspects specified in Article 17;

(e) a description of the planned conformity assessment procedure; 

(f) a written declaration that no other application has been filed with another notified body for the same application. 

This includes both the QMS itself and the specific AI systems within it, ensuring compliance with the requirements outlined in the context.

```
**Question 5:**
```text
 Answer this question:
What is the purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 13 June 2024

Answer:
WITH CHUNK SIZE 512 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 42/42 [00:02<00:00, 28.79it/s]Rebuilt: 1339 chunks, chunk_size=512, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.733]
.5.2016, p. 1).
Regulation (EU) 2018/1725 of the European Parliament and of the Council of 23 October 2018 on the protection of natural
persons with regard to the processing of personal data by the Union institutions, bodies, offices and agencies and on the free
movement of such data, and repealing Regulation (EC) No 45/2001 and Decision No 1247/2002/EC (OJ L 295, 21.11.2018, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.729]
Regulation (EU) No 1025/2012 of the European Parliament and of the Council of 25 October 2012 on European standardisation,
amending Council Directives 89/686/EEC and 93/15/EEC and Directives 94/9/EC, 94/25/EC, 95/16/EC, 97/23/EC, 98/34/EC,
2004/22/EC, 2007/23/EC, 2009/23/EC and 2009/105/EC of the European Parliament and of the Council and repealing Council
Decision 87/95/EEC and Decision No 1673/2006/EC of the European Parliament and of the Council (OJ L 316, 14.11.2012, p. 12).

---

[Source: EU_AI_Act.txt, Relevance: 0.721]
1).
(b) Regulation (EU) 2018/1861 of the European Parliament and of the Council of 28 November 2018 on the
establishment, operation and use of the Schengen Information System (SIS) in the field of border checks, and
amending the Convention implementing the Schengen Agreement, and amending and repealing Regulation (EC)
No 1987/2006 (OJ L 312, 7.12.2018, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.719]
14).
(c) Regulation (EU) 2018/1862 of the European Parliament and of the Council of 28 November 2018 on the
establishment, operation and use of the Schengen Information System (SIS) in the field of police cooperation and
judicial cooperation in criminal matters, amending and repealing Council Decision 2007/533/JHA, and
repealing Regulation (EC) No 1986/2006 of the European Parliament and of the Council and Commission
Decision 2010/261/EU (OJ L 312, 7.12.2018, p. 56).

2.

---

[Source: EU_AI_Act.txt, Relevance: 0.714]
for law enforcement purposes, amending Regulations
(EU) 2018/1240 and (EU) 2019/818 of the European Parliament and of the Council and repealing Regulation (EU)
No 603/2013 of the European Parliament and of the Council (OJ L, 2024/1358, 22.5.2024, ELI: http://data.europa.
eu/eli/reg/2024/1358/oj).

4.
============================================================

The purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 13 June 2024 is not explicitly stated in the given context. However, it appears to be related to the establishment, operation, and use of the Schengen Information System for law enforcement purposes, as indicated in the previous regulations mentioned. To provide more specific details about its exact purpose would require additional information from official sources or legal documents. Based solely on the provided context, we cannot determine the precise function of this regulation without further research.

```
### Chunk Size: 512
**Question 1:**
```text
 Answer this question:
Under Visa Information System, when was Regulation (EU) 2021/1133 adopted?

Answer:
WITH CHUNK SIZE 2048 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 11/11 [00:02<00:00,  5.12it/s]Rebuilt: 345 chunks, chunk_size=2048, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.615]
Schengen Information System
(a) Regulation (EU) 2018/1860 of the European Parliament and of the Council of 28 November 2018 on the use of
the Schengen Information System for the return of illegally staying third-country nationals (OJ L 312,
7.12.2018, p. 1).
(b) Regulation (EU) 2018/1861 of the European Parliament and of the Council of 28 November 2018 on the
establishment, operation and use of the Schengen Information System (SIS) in the field of border checks, and
amending the Convention implementing the Schengen Agreement, and amending and repealing Regulation (EC)
No 1987/2006 (OJ L 312, 7.12.2018, p. 14).
(c) Regulation (EU) 2018/1862 of the European Parliament and of the Council of 28 November 2018 on the
establishment, operation and use of the Schengen Information System (SIS) in the field of police cooperation and
judicial cooperation in criminal matters, amending and repealing Council Decision 2007/533/JHA, and
repealing Regulation (EC) No 1986/2006 of the European Parliament and of the Council and Commission
Decision 2010/261/EU (OJ L 312, 7.12.2018, p. 56).

2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.612]
European Criminal Records Information System on third-country nationals and stateless persons
Regulation (EU) 2019/816 of the European Parliament and of the Council of 17 April 2019 establishing
a centralised system for the identification of Member States holding conviction information on third-country
nationals and stateless persons (ECRIS-TCN) to supplement the European Criminal Records Information System and
amending Regulation (EU) 2018/1726 (OJ L 135, 22.5.2019, p. 1).

7.

Interoperability
(a) Regulation (EU) 2019/817 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767/2008, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1726 and (EU)
2018/1861 of the European Parliament and of the Council and Council Decisions 2004/512/EC and
2008/633/JHA (OJ L 135, 22.5.2019, p. 27).
(b) Regulation (EU) 2019/818 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of police and judicial cooperation,
asylum and migration and amending Regulations (EU) 2018/1726, (EU) 2018/1862 and (EU) 2019/816 (OJ
L 135, 22.5.2019, p. 85).

140/144

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

OJ L, 12.7.2024
ANNEX XI

Technical documentation referred to in Article 53(1), point (a) — technical documentation for
providers of general-purpose AI models

Section 1
Information to be provided by all providers of general-purpose AI models
The technical documentation referred to in Article 53(1), point (a) shall contain at least the following information as
appropriate to the size and risk profile of the model:
1.

---

[Source: EU_AI_Act.txt, Relevance: 0.560]
Article 102
Amendment to Regulation (EC) No 300/2008
In Article 4(3) of Regulation (EC) No 300/2008, the following subparagraph is added:
‘When adopting detailed measures related to technical specifications and procedures for approval and use of security
equipment concerning Artificial Intelligence systems within the meaning of Regulation (EU) 2024/1689 of the European
Parliament and of the Council (*), the requirements set out in Chapter III, Section 2, of that Regulation shall be taken into
account.
(*)

Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised
rules on artificial intelligence and amending Regulations (EC) No 300/2008, (EU) No 167/2013, (EU) No 168/2013,
(EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and Directives 2014/90/EU, (EU) 2016/797 and (EU)
2020/1828 (Artificial Intelligence Act) (OJ L, 2024/1689, 12.7.2024, ELI: http://data.europa.eu/eli/reg/
2024/1689/oj).’.

Article 103
Amendment to Regulation (EU) No 167/2013
In Article 17(5) of Regulation (EU) No 167/2013, the following subparagraph is added:
‘When adopting delegated acts pursuant to the first subparagraph concerning artificial intelligence systems which are safety
components within the meaning of Regulation (EU) 2024/1689 of the European Parliament and of the Council (*), the
requirements set out in Chapter III, Section 2, of that Regulation shall be taken into account.
(*)

118/144

Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised
rules on artificial intelligence and amending Regulations (EC) No 300/2008, (EU) No 167/2013, (EU) No 168/2013,
(EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and Directives 2014/90/EU, (EU) 2016/797 and (EU)
2020/1828 (Artificial Intelligence Act) (OJ L, 2024/1689, 12.7.2024, ELI: http://data.europa.eu/eli/reg/
2024/1689/oj).’.

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

OJ L, 12.7.2024
Article 104
Amendment to Regulation (EU) No 168/2013

---

[Source: EU_AI_Act.txt, Relevance: 0.560]
Article 108
Amendments to Regulation (EU) 2018/1139
Regulation (EU) 2018/1139 is amended as follows:
(1) in Article 17, the following paragraph is added:
‘3.
Without prejudice to paragraph 2, when adopting implementing acts pursuant to paragraph 1 concerning
Artificial Intelligence systems which are safety components within the meaning of Regulation (EU) 2024/1689 of the
European Parliament and of the Council (*), the requirements set out in Chapter III, Section 2, of that Regulation shall be
taken into account.
(*)

Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down
harmonised rules on artificial intelligence and amending Regulations (EC) No 300/2008, (EU) No 167/2013, (EU)
No 168/2013, (EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and Directives 2014/90/EU, (EU) 2016/797
and (EU) 2020/1828 (Artificial Intelligence Act) (OJ L, 2024/1689, 12.7.2024, ELI: http://data.europa.
eu/eli/reg/2024/1689/oj).’;

(2) in Article 19, the following paragraph is added:
‘4.
When adopting delegated acts pursuant to paragraphs 1 and 2 concerning Artificial Intelligence systems which
are safety components within the meaning of Regulation (EU) 2024/1689, the requirements set out in Chapter III,
Section 2, of that Regulation shall be taken into account.’;
(3) in Article 43, the following paragraph is added:
‘4.
When adopting implementing acts pursuant to paragraph 1 concerning Artificial Intelligence systems which are
safety components within the meaning of Regulation (EU) 2024/1689, the requirements set out in Chapter III,
Section 2, of that Regulation shall be taken into account.’;
(4) in Article 47, the following paragraph is added:
‘3.
When adopting delegated acts pursuant to paragraphs 1 and 2 concerning Artificial Intelligence systems which
are safety components within the meaning of Regulation (EU) 2024/1689, the requirements set out in Chapter III,
Section 2, of that Regulation shall be taken into account.’;
(5) in Article 57, the following subparagraph is added:
‘When adopt

---

[Source: EU_AI_Act.txt, Relevance: 0.558]
Regulation (EU) 2017/745 of the European Parliament and of the Council of 5 April 2017 on medical devices,
amending Directive 2001/83/EC, Regulation (EC) No 178/2002 and Regulation (EC) No 1223/2009 and repealing
Council Directives 90/385/EEC and 93/42/EEC (OJ L 117, 5.5.2017, p. 1);

12.

Regulation (EU) 2017/746 of the European Parliament and of the Council of 5 April 2017 on in vitro diagnostic
medical devices and repealing Directive 98/79/EC and Commission Decision 2010/227/EU (OJ L 117, 5.5.2017,
p. 176).
Section B. List of other Union harmonisation legislation

13.

Regulation (EC) No 300/2008 of the European Parliament and of the Council of 11 March 2008 on common rules
in the field of civil aviation security and repealing Regulation (EC) No 2320/2002 (OJ L 97, 9.4.2008, p. 72);

14.

Regulation (EU) No 168/2013 of the European Parliament and of the Council of 15 January 2013 on the approval
and market surveillance of two- or three-wheel vehicles and quadricycles (OJ L 60, 2.3.2013, p. 52);

15.

Regulation (EU) No 167/2013 of the European Parliament and of the Council of 5 February 2013 on the approval
and market surveillance of agricultural and forestry vehicles (OJ L 60, 2.3.2013, p. 1);

124/144

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

OJ L, 12.7.2024
16.

Directive 2014/90/EU of the European Parliament and of the Council of 23 July 2014 on marine equipment and
repealing Council Directive 96/98/EC (OJ L 257, 28.8.2014, p. 146);

17.

Directive (EU) 2016/797 of the European Parliament and of the Council of 11 May 2016 on the interoperability of
the rail system within the European Union (OJ L 138, 26.5.2016, p. 44);

18.

Regulation (EU) 2018/858 of the European Parliament and of the Council of 30 May 2018 on the approval and
market surveillance of motor vehicles and their trailers, and of systems, components and separate technical units
intended for such vehicles, amending Regulations (EC) No 715/2007 and (EC) No 595/2009 and repealing Directive
2007/46/EC (OJ L 151, 14.6.2018, p. 1);
============================================================

Regulation (EU) 2021/1133 was adopted on 7 July 2021. This regulation amends several existing regulations regarding the Visa Information System, including those mentioned in the given text. The exact details of its amendments can be found in the annexes of this document. 

This regulation addresses the establishment of conditions for accessing other EU information systems for the purposes of the Visa Information System. It aims to ensure interoperability among various EU information systems used in the context of visas and immigration. The adoption date aligns with the overall legislative process described in the provided context, indicating it was part of ongoing reforms aimed at enhancing the efficiency and effectiveness of the Visa Information System.

```
**Question 2:**
```text
 Answer this question:
Under Visa Information System, what is the Regulation number listed in point (b), and what is its stated purpose?

Answer:
WITH CHUNK SIZE 2048 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 11/11 [00:01<00:00,  8.53it/s]Rebuilt: 345 chunks, chunk_size=2048, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.474]
The CE marking shall be subject to the general principles set out in Article 30 of Regulation (EC) No 765/2008.

2.
For high-risk AI systems provided digitally, a digital CE marking shall be used, only if it can easily be accessed via the
interface from which that system is accessed or via an easily accessible machine-readable code or other electronic means.
3.
The CE marking shall be affixed visibly, legibly and indelibly for high-risk AI systems. Where that is not possible or
not warranted on account of the nature of the high-risk AI system, it shall be affixed to the packaging or to the
accompanying documentation, as appropriate.
4.
Where applicable, the CE marking shall be followed by the identification number of the notified body responsible for
the conformity assessment procedures set out in Article 43. The identification number of the notified body shall be affixed
by the body itself or, under its instructions, by the provider or by the provider’s authorised representative. The identification
number shall also be indicated in any promotional material which mentions that the high-risk AI system fulfils the
requirements for CE marking.
5.
Where high-risk AI systems are subject to other Union law which also provides for the affixing of the CE marking, the
CE marking shall indicate that the high-risk AI system also fulfil the requirements of that other law.

---

[Source: EU_AI_Act.txt, Relevance: 0.457]
References to any relevant harmonised standards used or any other common specification in relation to which
conformity is declared;

7.

Where applicable, the name and identification number of the notified body, a description of the conformity
assessment procedure performed, and identification of the certificate issued;

8.

The place and date of issue of the declaration, the name and function of the person who signed it, as well as an
indication for, or on behalf of whom, that person signed, a signature.

132/144

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

OJ L, 12.7.2024
ANNEX VI

Conformity assessment procedure based on internal control

1.

The conformity assessment procedure based on internal control is the conformity assessment procedure based on
points 2, 3 and 4.

2.

The provider verifies that the established quality management system is in compliance with the requirements of
Article 17.

3.

The provider examines the information contained in the technical documentation in order to assess the compliance
of the AI system with the relevant essential requirements set out in Chapter III, Section 2.

4.

The provider also verifies that the design and development process of the AI system and its post-market monitoring
as referred to in Article 72 is consistent with the technical documentation.

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

133/144

EN

OJ L, 12.7.2024
ANNEX VII

Conformity based on an assessment of the quality management system and an assessment of the
technical documentation

1.

Introduction
Conformity based on an assessment of the quality management system and an assessment of the technical
documentation is the conformity assessment procedure based on points 2 to 5.

2.

---

[Source: EU_AI_Act.txt, Relevance: 0.449]
Article 35
Identification numbers and lists of notified bodies
1.
The Commission shall assign a single identification number to each notified body, even where a body is notified under
more than one Union act.
2.
The Commission shall make publicly available the list of the bodies notified under this Regulation, including their
identification numbers and the activities for which they have been notified. The Commission shall ensure that the list is kept
up to date.

Article 36
Changes to notifications
1.
The notifying authority shall notify the Commission and the other Member States of any relevant changes to the
notification of a notified body via the electronic notification tool referred to in Article 30(2).
2.

The procedures laid down in Articles 29 and 30 shall apply to extensions of the scope of the notification.

For changes to the notification other than extensions of its scope, the procedures laid down in paragraphs (3) to (9) shall
apply.
ELI: http://data.europa.eu/eli/reg/2024/1689/oj

73/144

EN

OJ L, 12.7.2024
3.
Where a notified body decides to cease its conformity assessment activities, it shall inform the notifying authority and
the providers concerned as soon as possible and, in the case of a planned cessation, at least one year before ceasing its
activities. The certificates of the notified body may remain valid for a period of nine months after cessation of the notified
body’s activities, on condition that another notified body has confirmed in writing that it will assume responsibilities for the
high-risk AI systems covered by those certificates. The latter notified body shall complete a full assessment of the high-risk
AI systems affected by the end of that nine-month-period before issuing new certificates for those systems.

---

[Source: EU_AI_Act.txt, Relevance: 0.442]
On the basis of
the findings, that report shall, where appropriate, be accompanied by a proposal for amendment of this Regulation. The
reports shall be made public.
4.

The reports referred to in paragraph 2 shall pay specific attention to the following:

(a) the status of the financial, technical and human resources of the national competent authorities in order to effectively
perform the tasks assigned to them under this Regulation;
(b) the state of penalties, in particular administrative fines as referred to in Article 99(1), applied by Member States for
infringements of this Regulation;
(c) adopted harmonised standards and common specifications developed to support this Regulation;
(d) the number of undertakings that enter the market after the entry into application of this Regulation, and how many of
them are SMEs.
5.
By 2 August 2028, the Commission shall evaluate the functioning of the AI Office, whether the AI Office has been
given sufficient powers and competences to fulfil its tasks, and whether it would be relevant and needed for the proper
implementation and enforcement of this Regulation to upgrade the AI Office and its enforcement competences and to
increase its resources. The Commission shall submit a report on its evaluation to the European Parliament and to the
Council.
6.
By 2 August 2028 and every four years thereafter, the Commission shall submit a report on the review of the progress
on the development of standardisation deliverables on the energy-efficient development of general-purpose AI models, and
asses the need for further measures or actions, including binding measures or actions.

---

[Source: EU_AI_Act.txt, Relevance: 0.441]
When setting the functional
specifications of such database, the Commission shall consult the relevant experts, and when updating the functional
specifications of such database, the Commission shall consult the Board.
2.
The data listed in Sections A and B of Annex VIII shall be entered into the EU database by the provider or, where
applicable, by the authorised representative.
3.
The data listed in Section C of Annex VIII shall be entered into the EU database by the deployer who is, or who acts on
behalf of, a public authority, agency or body, in accordance with Article 49(3) and (4).
4.
With the exception of the section referred to in Article 49(4) and Article 60(4), point (c), the information contained in
the EU database registered in accordance with Article 49 shall be accessible and publicly available in a user-friendly manner.
The information should be easily navigable and machine-readable. The information registered in accordance with Article 60
shall be accessible only to market surveillance authorities and the Commission, unless the prospective provider or provider
has given consent for also making the information accessible the public.
5.
The EU database shall contain personal data only in so far as necessary for collecting and processing information in
accordance with this Regulation. That information shall include the names and contact details of natural persons who are
responsible for registering the system and have the legal authority to represent the provider or the deployer, as applicable.
100/144

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

OJ L, 12.7.2024
6.
The Commission shall be the controller of the EU database. It shall make available to providers, prospective providers
and deployers adequate technical and administrative support. The EU database shall comply with the applicable accessibility
requirements.

CHAPTER IX
POST-MARKET MONITORING, INFORMATION SHARING AND MARKET SURVEILLANCE

SECTION 1

Post-market monitoring
============================================================

The Regulation number listed in point (b) is 133/144, and its stated purpose is "Conformity based on an assessment of the quality management system and an assessment of the technical documentation". This regulation outlines the conformity assessment procedure that involves verifying the quality management system and assessing the technical documentation of high-risk AI systems. 

This answer is derived directly from the text provided, specifically from the last sentence of the given context, which states: "Conformity based on an assessment of the quality management system and an assessment of the technical documentation is the conformity assessment procedure based on points 2 to 5." The next sentence then identifies the reference document for this procedure, which is numbered 133/144. Additionally, the instruction asks about the purpose of this regulation, which is clearly stated in the same sentence as the reference number. Therefore, there's no ambiguity or lack of information required to provide a precise answer based solely on the context provided.

```
**Question 3:**
```text
 Answer this question:
Under Visa Information System, which Regulation establishes conditions for accessing other EU information systems, and which Regulation reforms the Visa Information System?

Answer:
WITH CHUNK SIZE 2048 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 11/11 [00:02<00:00,  5.90it/s]Rebuilt: 345 chunks, chunk_size=2048, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.628]
Schengen Information System
(a) Regulation (EU) 2018/1860 of the European Parliament and of the Council of 28 November 2018 on the use of
the Schengen Information System for the return of illegally staying third-country nationals (OJ L 312,
7.12.2018, p. 1).
(b) Regulation (EU) 2018/1861 of the European Parliament and of the Council of 28 November 2018 on the
establishment, operation and use of the Schengen Information System (SIS) in the field of border checks, and
amending the Convention implementing the Schengen Agreement, and amending and repealing Regulation (EC)
No 1987/2006 (OJ L 312, 7.12.2018, p. 14).
(c) Regulation (EU) 2018/1862 of the European Parliament and of the Council of 28 November 2018 on the
establishment, operation and use of the Schengen Information System (SIS) in the field of police cooperation and
judicial cooperation in criminal matters, amending and repealing Council Decision 2007/533/JHA, and
repealing Regulation (EC) No 1986/2006 of the European Parliament and of the Council and Commission
Decision 2010/261/EU (OJ L 312, 7.12.2018, p. 56).

2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.623]
European Criminal Records Information System on third-country nationals and stateless persons
Regulation (EU) 2019/816 of the European Parliament and of the Council of 17 April 2019 establishing
a centralised system for the identification of Member States holding conviction information on third-country
nationals and stateless persons (ECRIS-TCN) to supplement the European Criminal Records Information System and
amending Regulation (EU) 2018/1726 (OJ L 135, 22.5.2019, p. 1).

7.

Interoperability
(a) Regulation (EU) 2019/817 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767/2008, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1726 and (EU)
2018/1861 of the European Parliament and of the Council and Council Decisions 2004/512/EC and
2008/633/JHA (OJ L 135, 22.5.2019, p. 27).
(b) Regulation (EU) 2019/818 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of police and judicial cooperation,
asylum and migration and amending Regulations (EU) 2018/1726, (EU) 2018/1862 and (EU) 2019/816 (OJ
L 135, 22.5.2019, p. 85).

140/144

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

OJ L, 12.7.2024
ANNEX XI

Technical documentation referred to in Article 53(1), point (a) — technical documentation for
providers of general-purpose AI models

Section 1
Information to be provided by all providers of general-purpose AI models
The technical documentation referred to in Article 53(1), point (a) shall contain at least the following information as
appropriate to the size and risk profile of the model:
1.

---

[Source: EU_AI_Act.txt, Relevance: 0.595]
Regulation insofar as those systems are used for the purposes of law enforcement, migration, asylum and border
control management, or the administration of justice and democratic processes, should have effective investigative
and corrective powers, including at least the power to obtain access to all personal data that are being processed and
to all information necessary for the performance of its tasks. The market surveillance authorities should be able to
exercise their powers by acting with complete independence. Any limitations of their access to sensitive operational
data under this Regulation should be without prejudice to the powers conferred to them by Directive
(EU) 2016/680. No exclusion on disclosing data to national data protection authorities under this Regulation should
affect the current or future powers of those authorities beyond the scope of this Regulation.

(160) The market surveillance authorities and the Commission should be able to propose joint activities, including joint

investigations, to be conducted by market surveillance authorities or market surveillance authorities jointly with the
Commission, that have the aim of promoting compliance, identifying non-compliance, raising awareness and
providing guidance in relation to this Regulation with respect to specific categories of high-risk AI systems that are
found to present a serious risk across two or more Member States. Joint activities to promote compliance should be
carried out in accordance with Article 9 of Regulation (EU) 2019/1020. The AI Office should provide coordination
support for joint investigations.

(161) It is necessary to clarify the responsibilities and competences at Union and national level as regards AI systems that

are built on general-purpose AI models. To avoid overlapping competences, where an AI system is based on
a general-purpose AI model and the model and system are provided by the same provider, the supervision should
(49)
(50)
(51)
(52)

40/144

---

[Source: EU_AI_Act.txt, Relevance: 0.595]
The Commission may develop initiatives,
including of a sectoral nature, to facilitate the lowering of technical barriers hindering cross-border exchange of data
for AI development, including on data access infrastructure, semantic and technical interoperability of different types
of data.

(166) It is important that AI systems related to products that are not high-risk in accordance with this Regulation and thus

are not required to comply with the requirements set out for high-risk AI systems are nevertheless safe when placed
on the market or put into service. To contribute to this objective, Regulation (EU) 2023/988 of the European
Parliament and of the Council (53) would apply as a safety net.

(167) In order to ensure trustful and constructive cooperation of competent authorities on Union and national level, all

parties involved in the application of this Regulation should respect the confidentiality of information and data
obtained in carrying out their tasks, in accordance with Union or national law. They should carry out their tasks and
activities in such a manner as to protect, in particular, intellectual property rights, confidential business information
and trade secrets, the effective implementation of this Regulation, public and national security interests, the integrity
of criminal and administrative proceedings, and the integrity of classified information.

(168) Compliance with this Regulation should be enforceable by means of the imposition of penalties and other

---

[Source: EU_AI_Act.txt, Relevance: 0.591]
89).
Directive 2002/58/EC of the European Parliament and of the Council of 12 July 2002 concerning the processing of personal data
and the protection of privacy in the electronic communications sector (Directive on privacy and electronic communications) (OJ
L 201, 31.7.2002, p. 37).

(8)
(9)
(10)
(11)
(12)
(13)

(14)

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

3/144

EN

OJ L, 12.7.2024
rights and guarantees awarded to them by such Union law, including the rights related to solely automated individual
decision-making, including profiling. Harmonised rules for the placing on the market, the putting into service and
the use of AI systems established under this Regulation should facilitate the effective implementation and enable the
exercise of the data subjects’ rights and other remedies guaranteed under Union law on the protection of personal
data and of other fundamental rights.

4/144

(11)

This Regulation should be without prejudice to the provisions regarding the liability of providers of intermediary
services as set out in Regulation (EU) 2022/2065 of the European Parliament and of the Council (15).

(12)
============================================================

Under the Visa Information System, Regulation (EU) 2021/1133 amends Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards the establishment of the conditions for accessing other EU information systems for the purposes of the Visa Information System. Regulation (EU) 2021/1134 amends Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the Council and repeals Council Decisions 2004/512/EC and 2008/633/JHA for the purpose of reforming the Visa Information System.

```
**Question 4:**
```text
 Answer this question:
For the Quality management system, what should the application of the provider include?

Answer:
WITH CHUNK SIZE 2048 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 11/11 [00:01<00:00,  9.68it/s]Rebuilt: 345 chunks, chunk_size=2048, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.673]
Overview
The approved quality management system for the design, development and testing of AI systems pursuant to
Article 17 shall be examined in accordance with point 3 and shall be subject to surveillance as specified in point 5.
The technical documentation of the AI system shall be examined in accordance with point 4.

3.

Quality management system

3.1.

The application of the provider shall include:
(a) the name and address of the provider and, if the application is lodged by an authorised representative, also their
name and address;
(b) the list of AI systems covered under the same quality management system;
(c) the technical documentation for each AI system covered under the same quality management system;
(d) the documentation concerning the quality management system which shall cover all the aspects listed under
Article 17;
(e) a description of the procedures in place to ensure that the quality management system remains adequate and
effective;
(f) a written declaration that the same application has not been lodged with any other notified body.

3.2.

The quality management system shall be assessed by the notified body, which shall determine whether it satisfies the
requirements referred to in Article 17.
The decision shall be notified to the provider or its authorised representative.
The notification shall contain the conclusions of the assessment of the quality management system and the reasoned
assessment decision.

3.3.

The quality management system as approved shall continue to be implemented and maintained by the provider so
that it remains adequate and efficient.

3.4.

---

[Source: EU_AI_Act.txt, Relevance: 0.576]
the handling of communication with national competent authorities, other relevant authorities, including those
providing or supporting the access to data, notified bodies, other operators, customers or other interested parties;

(k) systems and procedures for record-keeping of all relevant documentation and information;
(l)

resource management, including security-of-supply related measures;

(m) an accountability framework setting out the responsibilities of the management and other staff with regard to all the
aspects listed in this paragraph.
2.
The implementation of the aspects referred to in paragraph 1 shall be proportionate to the size of the provider’s
organisation. Providers shall, in any event, respect the degree of rigour and the level of protection required to ensure the
compliance of their high-risk AI systems with this Regulation.
3.
Providers of high-risk AI systems that are subject to obligations regarding quality management systems or an
equivalent function under relevant sectoral Union law may include the aspects listed in paragraph 1 as part of the quality
management systems pursuant to that law.
4.
For providers that are financial institutions subject to requirements regarding their internal governance, arrangements
or processes under Union financial services law, the obligation to put in place a quality management system, with the
exception of paragraph 1, points (g), (h) and (i) of this Article, shall be deemed to be fulfilled by complying with the rules on
internal governance arrangements or processes pursuant to the relevant Union financial services law. To that end, any
harmonised standards referred to in Article 40 shall be taken into account.

---

[Source: EU_AI_Act.txt, Relevance: 0.556]
Any intended change to the approved quality management system or the list of AI systems covered by the latter shall
be brought to the attention of the notified body by the provider.
The proposed changes shall be examined by the notified body, which shall decide whether the modified quality
management system continues to satisfy the requirements referred to in point 3.2 or whether a reassessment is
necessary.
The notified body shall notify the provider of its decision. The notification shall contain the conclusions of the
examination of the changes and the reasoned assessment decision.

4.

Control of the technical documentation.

4.1.

In addition to the application referred to in point 3, an application with a notified body of their choice shall be
lodged by the provider for the assessment of the technical documentation relating to the AI system which the
provider intends to place on the market or put into service and which is covered by the quality management system
referred to under point 3.

4.2.

The application shall include:
(a) the name and address of the provider;
(b) a written declaration that the same application has not been lodged with any other notified body;
(c) the technical documentation referred to in Annex IV.

134/144

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

OJ L, 12.7.2024
4.3.

The technical documentation shall be examined by the notified body. Where relevant, and limited to what is
necessary to fulfil its tasks, the notified body shall be granted full access to the training, validation, and testing data
sets used, including, where appropriate and subject to security safeguards, through API or other relevant technical
means and tools enabling remote access.

4.4.

---

[Source: EU_AI_Act.txt, Relevance: 0.547]
The provider should establish a sound quality management system, ensure the accomplishment of the required
conformity assessment procedure, draw up the relevant documentation and establish a robust post-market
monitoring system. Providers of high-risk AI systems that are subject to obligations regarding quality management
systems under relevant sectoral Union law should have the possibility to include the elements of the quality
management system provided for in this Regulation as part of the existing quality management system provided for
in that other sectoral Union law. The complementarity between this Regulation and existing sectoral Union law
should also be taken into account in future standardisation activities or guidance adopted by the Commission. Public
authorities which put into service high-risk AI systems for their own use may adopt and implement the rules for the
quality management system as part of the quality management system adopted at a national or regional level, as
appropriate, taking into account the specificities of the sector and the competences and organisation of the public
authority concerned.

(82)

To enable enforcement of this Regulation and create a level playing field for operators, and, taking into account the
different forms of making available of digital products, it is important to ensure that, under all circumstances,
a person established in the Union can provide authorities with all the necessary information on the compliance of an
AI system. Therefore, prior to making their AI systems available in the Union, providers established in third
countries should, by written mandate, appoint an authorised representative established in the Union. This authorised
representative plays a pivotal role in ensuring the compliance of the high-risk AI systems placed on the market or
put into service in the Union by those providers who are not established in the Union and in serving as their contact
person established in the Union.

(83)

---

[Source: EU_AI_Act.txt, Relevance: 0.442]
Any change to the AI system that could affect the compliance of the AI system with the requirements or its intended
purpose shall be assessed by the notified body which issued the Union technical documentation assessment
certificate. The provider shall inform such notified body of its intention to introduce any of the abovementioned
changes, or if it otherwise becomes aware of the occurrence of such changes. The intended changes shall be assessed
by the notified body, which shall decide whether those changes require a new conformity assessment in accordance
with Article 43(4) or whether they could be addressed by means of a supplement to the Union technical
documentation assessment certificate. In the latter case, the notified body shall assess the changes, notify the
provider of its decision and, where the changes are approved, issue to the provider a supplement to the Union
technical documentation assessment certificate.

5.

Surveillance of the approved quality management system.

5.1.

The purpose of the surveillance carried out by the notified body referred to in Point 3 is to make sure that the
provider duly complies with the terms and conditions of the approved quality management system.

5.2.

For assessment purposes, the provider shall allow the notified body to access the premises where the design,
development, testing of the AI systems is taking place. The provider shall further share with the notified body all
necessary information.

5.3.

The notified body shall carry out periodic audits to make sure that the provider maintains and applies the quality
management system and shall provide the provider with an audit report. In the context of those audits, the notified
body may carry out additional tests of the AI systems for which a Union technical documentation assessment
certificate was issued.

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

135/144

EN

OJ L, 12.7.2024
ANNEX VIII

Information to be submitted upon the registration of high-risk AI systems in accordance with
Article 49
============================================================

According to the context, the application of the provider should include:

(a) the name and address of the provider and, if the application is lodged by an authorised representative, also their name and address;
(b) the list of AI systems covered under the same quality management system;
(c) the technical documentation for each AI system covered under the same quality management system;
(d) the documentation concerning the quality management system which shall cover all the aspects listed under Article 17;
(e) a description of the procedures in place to ensure that the quality management system remains adequate and effective;
(f) a written declaration that the same application has not been lodged with any other notified body. 

This includes details about the provider's identity, the AI systems involved, the technical documentation, the quality management system itself, and commitments to maintain the system's adequacy and effectiveness.

```
**Question 5:**
```text
 Answer this question:
What is the purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 13 June 2024

Answer:
WITH CHUNK SIZE 2048 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 11/11 [00:01<00:00,  8.82it/s]Rebuilt: 345 chunks, chunk_size=2048, chunk_overlap=0
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.743]
Official Journal
of the European Union

EN
L series

2024/1689

12.7.2024

REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
of 13 June 2024
laying down harmonised rules on artificial intelligence and amending Regulations (EC) No 300/2008,
(EU) No 167/2013, (EU) No 168/2013, (EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and
Directives 2014/90/EU, (EU) 2016/797 and (EU) 2020/1828 (Artificial Intelligence Act)
(Text with EEA relevance)

THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION,

Having regard to the Treaty on the Functioning of the European Union, and in particular Articles 16 and 114 thereof,
Having regard to the proposal from the European Commission,
After transmission of the draft legislative act to the national parliaments,
Having regard to the opinion of the European Economic and Social Committee (1),
Having regard to the opinion of the European Central Bank (2),
Having regard to the opinion of the Committee of the Regions (3),
Acting in accordance with the ordinary legislative procedure (4),
Whereas:
(1)

---

[Source: EU_AI_Act.txt, Relevance: 0.680]
OJ L, 12.7.2024
5.
As soon as it adopts a delegated act, the Commission shall notify it simultaneously to the European Parliament and to
the Council.
6.
Any delegated act adopted pursuant to Article 6(6) or (7), Article 7(1) or (3), Article 11(3), Article 43(5) or (6),
Article 47(5), Article 51(3), Article 52(4) or Article 53(5) or (6) shall enter into force only if no objection has been
expressed by either the European Parliament or the Council within a period of three months of notification of that act to
the European Parliament and the Council or if, before the expiry of that period, the European Parliament and the Council
have both informed the Commission that they will not object. That period shall be extended by three months at the
initiative of the European Parliament or of the Council.

Article 98
Committee procedure
1.
The Commission shall be assisted by a committee. That committee shall be a committee within the meaning of
Regulation (EU) No 182/2011.
2.

Where reference is made to this paragraph, Article 5 of Regulation (EU) No 182/2011 shall apply.

CHAPTER XII
PENALTIES

---

[Source: EU_AI_Act.txt, Relevance: 0.666]
Regulation (EU) 2017/745 of the European Parliament and of the Council of 5 April 2017 on medical devices,
amending Directive 2001/83/EC, Regulation (EC) No 178/2002 and Regulation (EC) No 1223/2009 and repealing
Council Directives 90/385/EEC and 93/42/EEC (OJ L 117, 5.5.2017, p. 1);

12.

Regulation (EU) 2017/746 of the European Parliament and of the Council of 5 April 2017 on in vitro diagnostic
medical devices and repealing Directive 98/79/EC and Commission Decision 2010/227/EU (OJ L 117, 5.5.2017,
p. 176).
Section B. List of other Union harmonisation legislation

13.

Regulation (EC) No 300/2008 of the European Parliament and of the Council of 11 March 2008 on common rules
in the field of civil aviation security and repealing Regulation (EC) No 2320/2002 (OJ L 97, 9.4.2008, p. 72);

14.

Regulation (EU) No 168/2013 of the European Parliament and of the Council of 15 January 2013 on the approval
and market surveillance of two- or three-wheel vehicles and quadricycles (OJ L 60, 2.3.2013, p. 52);

15.

Regulation (EU) No 167/2013 of the European Parliament and of the Council of 5 February 2013 on the approval
and market surveillance of agricultural and forestry vehicles (OJ L 60, 2.3.2013, p. 1);

124/144

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

OJ L, 12.7.2024
16.

Directive 2014/90/EU of the European Parliament and of the Council of 23 July 2014 on marine equipment and
repealing Council Directive 96/98/EC (OJ L 257, 28.8.2014, p. 146);

17.

Directive (EU) 2016/797 of the European Parliament and of the Council of 11 May 2016 on the interoperability of
the rail system within the European Union (OJ L 138, 26.5.2016, p. 44);

18.

Regulation (EU) 2018/858 of the European Parliament and of the Council of 30 May 2018 on the approval and
market surveillance of motor vehicles and their trailers, and of systems, components and separate technical units
intended for such vehicles, amending Regulations (EC) No 715/2007 and (EC) No 595/2009 and repealing Directive
2007/46/EC (OJ L 151, 14.6.2018, p. 1);

---

[Source: EU_AI_Act.txt, Relevance: 0.643]
Schengen Information System
(a) Regulation (EU) 2018/1860 of the European Parliament and of the Council of 28 November 2018 on the use of
the Schengen Information System for the return of illegally staying third-country nationals (OJ L 312,
7.12.2018, p. 1).
(b) Regulation (EU) 2018/1861 of the European Parliament and of the Council of 28 November 2018 on the
establishment, operation and use of the Schengen Information System (SIS) in the field of border checks, and
amending the Convention implementing the Schengen Agreement, and amending and repealing Regulation (EC)
No 1987/2006 (OJ L 312, 7.12.2018, p. 14).
(c) Regulation (EU) 2018/1862 of the European Parliament and of the Council of 28 November 2018 on the
establishment, operation and use of the Schengen Information System (SIS) in the field of police cooperation and
judicial cooperation in criminal matters, amending and repealing Council Decision 2007/533/JHA, and
repealing Regulation (EC) No 1986/2006 of the European Parliament and of the Council and Commission
Decision 2010/261/EU (OJ L 312, 7.12.2018, p. 56).

2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.620]
The Commission may develop initiatives,
including of a sectoral nature, to facilitate the lowering of technical barriers hindering cross-border exchange of data
for AI development, including on data access infrastructure, semantic and technical interoperability of different types
of data.

(166) It is important that AI systems related to products that are not high-risk in accordance with this Regulation and thus

are not required to comply with the requirements set out for high-risk AI systems are nevertheless safe when placed
on the market or put into service. To contribute to this objective, Regulation (EU) 2023/988 of the European
Parliament and of the Council (53) would apply as a safety net.

(167) In order to ensure trustful and constructive cooperation of competent authorities on Union and national level, all

parties involved in the application of this Regulation should respect the confidentiality of information and data
obtained in carrying out their tasks, in accordance with Union or national law. They should carry out their tasks and
activities in such a manner as to protect, in particular, intellectual property rights, confidential business information
and trade secrets, the effective implementation of this Regulation, public and national security interests, the integrity
of criminal and administrative proceedings, and the integrity of classified information.

(168) Compliance with this Regulation should be enforceable by means of the imposition of penalties and other
============================================================

The purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 13 June 2024 is to lay down harmonised rules on artificial intelligence and amending Regulations (EC) No 300/2008, (EU) No 167/2013, (EU) No 168/2013, (EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and Directives 2014/90/EU, (EU) 2016/797 and (EU) 2020/1828 (Artificial Intelligence Act). This regulation aims to provide guidelines and standards for the development, deployment, and use of artificial intelligence across the European Union. It seeks to harmonize existing regulations and directives related to artificial intelligence while ensuring consistency and effectiveness in the application of these rules throughout the region. Additionally, it includes provisions for penalties and other enforcement mechanisms to ensure compliance with the new regulatory framework. The regulation also emphasizes the importance of fostering trustful and constructive cooperation between competent authorities at both Union and national levels, respecting confidentiality and protecting various forms of sensitive information. Lastly, it mandates the development of initiatives aimed at facilitating the lowering of technical barriers hindering cross-border exchange of data for AI development.

```