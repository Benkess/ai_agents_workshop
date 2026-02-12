# Exercise 9: Retrieval Score Analysis
Analyze the similarity scores returned by your retrieval system. You can use any of the copora and your own queries.

**For 10 different queries:**

- Retrieve top 10 chunks

- Record all similarity scores

- Examine the score distribution

**Look for patterns:**

- When is there a clear "winner" (large gap between #1 and #2)?

- When are scores tightly clustered (ambiguous)?

- What score threshold would you use to filter out irrelevant results?

- How does score distribution correlate with answer quality?

**Experiment:** 

Implement a score threshold (e.g., only include chunks with score > 0.5). 

How does this affect results?

## Answers: 
### When is there a clear "winner" (large gap between #1 and #2)?
When I basically quote the source of the question like in question 10. But also not that question 10 retrieved a lot of high scores for irelivant chunks as well.

### When are scores tightly clustered (ambiguous)?
When no big or uncommon words are used and the question is less of a quote and more about high level senantics.

### What score threshold would you use to filter out irrelevant results?
Honestly there is not a great one. Sometimes irrelevant chunks get pretty high scores like 0.7 and sometimes relevant chunks get low scores like 0.4. I guess maybe 0.45.

### How does score distribution correlate with answer quality?
Not much. It kinda always gets it right.

### Threshold results
Setting 0.5 get rid of a lot of irrelicant stuff but sometimes the answer.

## Questions:
```text
question_1 = "Under Visa Information System, when was Regulation (EU) 2021/1133 adopted?" # ← Use a corpus-specific question!
question_2 = "Under Visa Information System, what is the Regulation number listed in point (b), and what is its stated purpose?" # ← Use a corpus-specific question!
question_3 = "Under Visa Information System, which Regulation establishes conditions for accessing other EU information systems, and which Regulation reforms the Visa Information System?" # ← Use a corpus-specific question!
question_4 = "For the Quality management system, what should the application of the provider include?" # ← Use a corpus-specific question!
question_5 = "What is the purpose of REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 13 June 2024" # ← Use a corpus-specific question!
question_6 = "How do I adjust the carburetor on a Model T?"
question_7 = "What is the correct spark plug gap for a Model T Ford?"
question_8 = "How do I fix a slipping transmission band?"
question_9 = "What oil should I use in a Model T engine?"
question_10 = "How should the notion of ‘biometric categorisation’, referred to in Regulation 2024/1689, be defined?"
```

Question 10 text:
```text
The notion of ‘biometric categorisation’ referred to in this Regulation should be defined as assigning natural persons
to specific categories on the basis of their biometric data. Such specific categories can relate to aspects such as sex,
age, hair colour, eye colour, tattoos, behavioural or personality traits, language, religion, membership of a national
minority, sexual or political orientation. This does not include biometric categorisation systems that are a purely
ancillary feature intrinsically linked to another commercial service, meaning that the feature cannot, for objective
technical reasons, be used without the principal service, and the integration of that feature or functionality is not
a means to circumvent the applicability of the rules of this Regulation. For example, filters categorising facial or body
features used on online marketplaces could constitute such an ancillary feature as they can be used only in relation to
the principal service which consists in selling a product by allowing the consumer to preview the display of the
product on him or herself and help the consumer to make a purchase decision. Filters used on online social network
services which categorise facial or body features to allow users to add or modify pictures or videos could also be
considered to be ancillary feature as such filter cannot be used without the principal service of the social network
services consisting in the sharing of content online.
```

## Results
### Question 1:
```text
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.779]
between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767/2008, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1726 and (EU)
2018/1861 of the European Parliament and of the Council and Council Decisions 2004/512/EC and
2008/633/JHA (OJ L 135, 22.5.2019, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.770]
/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and

---

[Source: EU_AI_Act.txt, Relevance: 0.757]
tters, amending and repealing Council Decision 2007/533/JHA, and
repealing Regulation (EC) No 1986/2006 of the European Parliament and of the Council and Commission
Decision 2010/261/EU (OJ L 312, 7.12.2018, p. 56).

2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU info

---

[Source: EU_AI_Act.txt, Relevance: 0.753]
(EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.667]
ding Regulations (EU)
No 1077/2011, (EU) No 515/2014, (EU) 2016/399, (EU) 2016/1624 and (EU) 2017/2226 (OJ L 236,
19.9.2018, p. 1).
(b) Regulation (EU) 2018/1241 of the European Parliament and of the Council of 12 September 2018 amending
Regulation (EU) 2016/794 for the purpose of establishing a European Travel Information and Authorisation
System (ETIAS) (OJ L 236, 19.9.2018, p. 72).

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

139/144

EN

OJ L, 12.7.2024
6.

---

[Source: EU_AI_Act.txt, Relevance: 0.660]
n implementing the Schengen Agreement and Regulations (EC)
No 767/2008 and (EU) No 1077/2011 (OJ L 327, 9.12.2017, p. 20).

5.

European Travel Information and Authorisation System
(a) Regulation (EU) 2018/1240 of the European Parliament and of the Council of 12 September 2018 establishing
a European Travel Information and Authorisation System (ETIAS) and amending Regulations (EU)
No 1077/2011, (EU) No 515/2014, (EU) 2016/399, (EU) 2016/1624 and (EU) 2017/2226 (OJ L 236,
19.9.2018, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.658]
/1861 of the European Parliament and of the Council and Council Decisions 2004/512/EC and
2008/633/JHA (OJ L 135, 22.5.2019, p. 27).
(b) Regulation (EU) 2019/818 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of police and judicial cooperation,
asylum and migration and amending Regulations (EU) 2018/1726, (EU) 2018/1862 and (EU) 2019/816 (OJ
L 135, 22.5.2019, p. 85).

140/144

---

[Source: EU_AI_Act.txt, Relevance: 0.644]
sions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

Eurodac
Regulation (EU) 2024/1358 of the European Parliament and of the Council of 14 May 2024 on the establishment of
‘Eurodac’ for the comparison of biometric data in order to effectively apply Regulations (EU) 2024/1315 and (EU)
2024/1350 of the European Parliament and of the Council and Council Directive 2001/55/EC and to identify
illegally staying third-country nationals and s

---

[Source: EU_AI_Act.txt, Relevance: 0.622]
of 30 May 2022 on European data governance and
amending Regulation (EU) 2018/1724 (Data Governance Act) (OJ L 152, 3.6.2022, p. 1).
Regulation (EU) 2023/2854 of the European Parliament and of the Council of 13 December 2023 on harmonised rules on fair
access to and use of data and amending Regulation (EU) 2017/2394 and Directive (EU) 2020/1828 (Data Act) (OJ L, 2023/2854,
22.12.2023, ELI: http://data.europa.eu/eli/reg/2023/2854/oj).

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

---

[Source: EU_AI_Act.txt, Relevance: 0.609]
EN

OJ L, 12.7.2024
ANNEX X
Union legislative acts on large-scale IT systems in the area of Freedom, Security and Justice

1.

Schengen Information System
(a) Regulation (EU) 2018/1860 of the European Parliament and of the Council of 28 November 2018 on the use of
the Schengen Information System for the return of illegally staying third-country nationals (OJ L 312,
7.12.2018, p.
============================================================

Context Scores:
[0.7792211771011353, 0.7695711851119995, 0.7574025988578796, 0.7532206177711487, 0.6671205759048462, 0.6601589918136597, 0.657646894454956, 0.6440409421920776, 0.6219672560691833, 0.6092253923416138]
============================================================


```
### Question 2:
```text
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.534]
tters, amending and repealing Council Decision 2007/533/JHA, and
repealing Regulation (EC) No 1986/2006 of the European Parliament and of the Council and Commission
Decision 2010/261/EU (OJ L 312, 7.12.2018, p. 56).

2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU info

---

[Source: EU_AI_Act.txt, Relevance: 0.534]
/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and

---

[Source: EU_AI_Act.txt, Relevance: 0.521]
(EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.519]
s Regulation;
(b) the state of penalties, in particular administrative fines as referred to in Article 99(1), applied by Member States for
infringements of this Regulation;
(c) adopted harmonised standards and common specifications developed to support this Regulation;
(d) the number of undertakings that enter the market after the entry into application of this Regulation, and how many of
them are SMEs.
5.
By 2 August 2028, the Commission shall evaluate the functioning of the AI Office, whether the AI Offic

---

[Source: EU_AI_Act.txt, Relevance: 0.514]
est, issuance and exercise of, as well as
supervision and reporting relating to, the authorisations referred to in paragraph 3. Those rules shall also specify in respect
of which of the objectives listed in paragraph 1, first subparagraph, point (h), including which of the criminal offences
referred to in point (h)(iii) thereof, the competent authorities may be authorised to use those systems for the purposes of
law enforcement.

---

[Source: EU_AI_Act.txt, Relevance: 0.497]
between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767/2008, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1726 and (EU)
2018/1861 of the European Parliament and of the Council and Council Decisions 2004/512/EC and
2008/633/JHA (OJ L 135, 22.5.2019, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.495]
ding Regulations (EU)
No 1077/2011, (EU) No 515/2014, (EU) 2016/399, (EU) 2016/1624 and (EU) 2017/2226 (OJ L 236,
19.9.2018, p. 1).
(b) Regulation (EU) 2018/1241 of the European Parliament and of the Council of 12 September 2018 amending
Regulation (EU) 2016/794 for the purpose of establishing a European Travel Information and Authorisation
System (ETIAS) (OJ L 236, 19.9.2018, p. 72).

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

139/144

EN

OJ L, 12.7.2024
6.

---

[Source: EU_AI_Act.txt, Relevance: 0.492]
the meaning of Regulation (EU) 2024/1689, the requirements set out in Chapter III,
Section 2, of that Regulation shall be taken into account.’;
(3) in Article 43, the following paragraph is added:
‘4.
When adopting implementing acts pursuant to paragraph 1 concerning Artificial Intelligence systems which are
safety components within the meaning of Regulation (EU) 2024/1689, the requirements set out in Chapter III,
Section 2, of that Regulation shall be taken into account.’;
(4) in Article 47, the following

---

[Source: EU_AI_Act.txt, Relevance: 0.490]
aph, point (h), (2) to (6) and Article 26(10) of this Regulation

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

11/144

EN

OJ L, 12.7.2024
adopted on the basis of Article 16 TFEU, or subject to their application, which relate to the processing of personal
data by the Member States when carrying out activities falling within the scope of Chapter 4 or Chapter 5 of
Title V of Part Three of the TFEU.

(42)

---

[Source: EU_AI_Act.txt, Relevance: 0.485]
plication of this Regulation, taking into account international approaches.
2.
The AI Office and the Board shall aim to ensure that the codes of practice cover at least the obligations provided for in
Articles 53 and 55, including the following issues:
86/144

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

EN

OJ L, 12.7.2024
(a) the means to ensure that the information referred to in Article 53(1), points (a) and (b), is kept up to date in light of
market and technological developments;
============================================================

Context Scores:
[0.5341213941574097, 0.5341200828552246, 0.5212445259094238, 0.5194149017333984, 0.5138257145881653, 0.4966033101081848, 0.4953490197658539, 0.4923830032348633, 0.48977914452552795, 0.4854663610458374]
============================================================



```
### Question 3:
```text
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.784]
tters, amending and repealing Council Decision 2007/533/JHA, and
repealing Regulation (EC) No 1986/2006 of the European Parliament and of the Council and Commission
Decision 2010/261/EU (OJ L 312, 7.12.2018, p. 56).

2.

Visa Information System
(a) Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU info

---

[Source: EU_AI_Act.txt, Relevance: 0.779]
/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards
the establishment of the conditions for accessing other EU information systems for the purposes of the Visa
Information System (OJ L 248, 13.7.2021, p. 1).
(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending
Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and

---

[Source: EU_AI_Act.txt, Relevance: 0.747]
between EU information systems in the field of borders and visa and amending
Regulations (EC) No 767/2008, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1726 and (EU)
2018/1861 of the European Parliament and of the Council and Council Decisions 2004/512/EC and
2008/633/JHA (OJ L 135, 22.5.2019, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.667]
(EU) 2017/2226, (EU) 2018/1240, (EU)
2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the
Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

---

[Source: EU_AI_Act.txt, Relevance: 0.647]
EN

OJ L, 12.7.2024
ANNEX X
Union legislative acts on large-scale IT systems in the area of Freedom, Security and Justice

1.

Schengen Information System
(a) Regulation (EU) 2018/1860 of the European Parliament and of the Council of 28 November 2018 on the use of
the Schengen Information System for the return of illegally staying third-country nationals (OJ L 312,
7.12.2018, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.628]
n implementing the Schengen Agreement and Regulations (EC)
No 767/2008 and (EU) No 1077/2011 (OJ L 327, 9.12.2017, p. 20).

5.

European Travel Information and Authorisation System
(a) Regulation (EU) 2018/1240 of the European Parliament and of the Council of 12 September 2018 establishing
a European Travel Information and Authorisation System (ETIAS) and amending Regulations (EU)
No 1077/2011, (EU) No 515/2014, (EU) 2016/399, (EU) 2016/1624 and (EU) 2017/2226 (OJ L 236,
19.9.2018, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.627]
ding Regulations (EU)
No 1077/2011, (EU) No 515/2014, (EU) 2016/399, (EU) 2016/1624 and (EU) 2017/2226 (OJ L 236,
19.9.2018, p. 1).
(b) Regulation (EU) 2018/1241 of the European Parliament and of the Council of 12 September 2018 amending
Regulation (EU) 2016/794 for the purpose of establishing a European Travel Information and Authorisation
System (ETIAS) (OJ L 236, 19.9.2018, p. 72).

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

139/144

EN

OJ L, 12.7.2024
6.

---

[Source: EU_AI_Act.txt, Relevance: 0.618]
the use of
the Schengen Information System for the return of illegally staying third-country nationals (OJ L 312,
7.12.2018, p. 1).
(b) Regulation (EU) 2018/1861 of the European Parliament and of the Council of 28 November 2018 on the
establishment, operation and use of the Schengen Information System (SIS) in the field of border checks, and
amending the Convention implementing the Schengen Agreement, and amending and repealing Regulation (EC)
No 1987/2006 (OJ L 312, 7.12.2018, p.

---

[Source: EU_AI_Act.txt, Relevance: 0.615]
sions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the
Visa Information System (OJ L 248, 13.7.2021, p. 11).

3.

Eurodac
Regulation (EU) 2024/1358 of the European Parliament and of the Council of 14 May 2024 on the establishment of
‘Eurodac’ for the comparison of biometric data in order to effectively apply Regulations (EU) 2024/1315 and (EU)
2024/1350 of the European Parliament and of the Council and Council Directive 2001/55/EC and to identify
illegally staying third-country nationals and s

---

[Source: EU_AI_Act.txt, Relevance: 0.611]
e to regulated financial institutions in the course of provision of those services, including when they make
use of AI systems. In order to ensure coherent application and enforcement of the obligations under this Regulation
and relevant rules and requirements of the Union financial services legal acts, the competent authorities for the
supervision and enforcement of those legal acts, in particular competent authorities as defined in Regulation (EU)
No 575/2013 of the European Parliament and of the Council
============================================================

Context Scores:
[0.7841933965682983, 0.7789629697799683, 0.7474085092544556, 0.6666969060897827, 0.6470073461532593, 0.6280574202537537, 0.6274077892303467, 0.6181766390800476, 0.615080714225769, 0.6113529801368713]
============================================================


```
### Question 4:
```text
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.686]
place on the market or put into service and which is covered by the quality management system
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

---

[Source: EU_AI_Act.txt, Relevance: 0.660]
is to make sure that the
provider duly complies with the terms and conditions of the approved quality management system.

5.2.

For assessment purposes, the provider shall allow the notified body to access the premises where the design,
development, testing of the AI systems is taking place. The provider shall further share with the notified body all
necessary information.

5.3.

---

[Source: EU_AI_Act.txt, Relevance: 0.616]
he technical documentation of the AI system shall be examined in accordance with point 4.

3.

Quality management system

3.1.

The application of the provider shall include:
(a) the name and address of the provider and, if the application is lodged by an authorised representative, also their
name and address;
(b) the list of AI systems covered under the same quality management system;
(c) the technical documentation for each AI system covered under the same quality management system;
(d) the documentation

---

[Source: EU_AI_Act.txt, Relevance: 0.601]
n shall contain the conclusions of the assessment of the quality management system and the reasoned
assessment decision.

3.3.

The quality management system as approved shall continue to be implemented and maintained by the provider so
that it remains adequate and efficient.

3.4.

---

[Source: EU_AI_Act.txt, Relevance: 0.600]
or other interested parties;

(k) systems and procedures for record-keeping of all relevant documentation and information;
(l)

resource management, including security-of-supply related measures;

(m) an accountability framework setting out the responsibilities of the management and other staff with regard to all the
aspects listed in this paragraph.
2.
The implementation of the aspects referred to in paragraph 1 shall be proportionate to the size of the provider’s
organisation.

---

[Source: EU_AI_Act.txt, Relevance: 0.594]
lity management systems or an
equivalent function under relevant sectoral Union law may include the aspects listed in paragraph 1 as part of the quality
management systems pursuant to that law.
4.
For providers that are financial institutions subject to requirements regarding their internal governance, arrangements
or processes under Union financial services law, the obligation to put in place a quality management system, with the
exception of paragraph 1, points (g), (h) and (i) of this Article, shall be d

---

[Source: EU_AI_Act.txt, Relevance: 0.587]
management system, in a simplified manner which would reduce the administrative burden and
the costs for those enterprises without affecting the level of protection and the need for compliance with the
requirements for high-risk AI systems. The Commission should develop guidelines to specify the elements of the
quality management system to be fulfilled in this simplified manner by microenterprises.

---

[Source: EU_AI_Act.txt, Relevance: 0.576]
m as approved shall continue to be implemented and maintained by the provider so
that it remains adequate and efficient.

3.4.

Any intended change to the approved quality management system or the list of AI systems covered by the latter shall
be brought to the attention of the notified body by the provider.
The proposed changes shall be examined by the notified body, which shall decide whether the modified quality
management system continues to satisfy the requirements referred to in point 3.2 or whether a

---

[Source: EU_AI_Act.txt, Relevance: 0.565]
tem;
(c) the technical documentation for each AI system covered under the same quality management system;
(d) the documentation concerning the quality management system which shall cover all the aspects listed under
Article 17;
(e) a description of the procedures in place to ensure that the quality management system remains adequate and
effective;
(f) a written declaration that the same application has not been lodged with any other notified body.

3.2.

---

[Source: EU_AI_Act.txt, Relevance: 0.535]
ment and of the Council (38) and Directive (EU) 2019/882. Providers should ensure
compliance with these requirements by design. Therefore, the necessary measures should be integrated as much as
possible into the design of the high-risk AI system.

(81)

The provider should establish a sound quality management system, ensure the accomplishment of the required
conformity assessment procedure, draw up the relevant documentation and establish a robust post-market
monitoring system.
============================================================

Context Scores:
[0.6857635974884033, 0.6603983640670776, 0.6162415146827698, 0.6010596752166748, 0.6003954410552979, 0.5943413972854614, 0.5865125060081482, 0.575626015663147, 0.5646910667419434, 0.5346402525901794]
============================================================


```
### Question 5:
```text
============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.744]
7/2013 of the European Parliament and of the Council (25), Regulation
(EU) No 168/2013 of the European Parliament and of the Council (26), Directive 2014/90/EU of the European
Parliament and of the Council (27), Directive (EU) 2016/797 of the European Parliament and of the Council (28),
Regulation (EU) 2018/858 of the European Parliament and of the Council (29), Regulation (EU) 2018/1139 of the

(24)

---

[Source: EU_AI_Act.txt, Relevance: 0.729]
y the Regulation (EC) No 810/2009 of the European Parliament and

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

17/144

EN

OJ L, 12.7.2024
of the Council (32), the Directive 2013/32/EU of the European Parliament and of the Council (33), and other relevant
Union law.

---

[Source: EU_AI_Act.txt, Relevance: 0.715]
on it to the European Parliament, the Council and the European Economic and Social Committee, taking into
account the first years of application of this Regulation. On the basis of the findings, that report shall, where appropriate, be
accompanied by a proposal for amendment of this Regulation with regard to the structure of enforcement and the need for
a Union agency to resolve any identified shortcomings.
Article 113
Entry into force and application
This Regulation shall enter into force on the twentieth

---

[Source: EU_AI_Act.txt, Relevance: 0.693]
2019/2144 and
Directives 2014/90/EU, (EU) 2016/797 and (EU) 2020/1828 (Artificial Intelligence Act)
(Text with EEA relevance)

THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION,

Having regard to the Treaty on the Functioning of the European Union, and in particular Articles 16 and 114 thereof,
Having regard to the proposal from the European Commission,
After transmission of the draft legislative act to the national parliaments,
Having regard to the opinion of the European Economic and Social C

---

[Source: EU_AI_Act.txt, Relevance: 0.690]
Article 78 shall apply from 2 August 2025, with the
exception of Article 101;
(c) Article 6(1) and the corresponding obligations in this Regulation shall apply from 2 August 2027.
This Regulation shall be binding in its entirety and directly applicable in all Member States.
Done at Brussels, 13 June 2024.
For the European Parliament

For the Council

The President

The President

R. METSOLA

M. MICHEL

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

123/144

EN

OJ L, 12.7.2024
ANNEX I

---

[Source: EU_AI_Act.txt, Relevance: 0.689]
mponents within the meaning of Regulation (EU) 2024/1689 of the European Parliament and of the Council (*),
the requirements set out in Chapter III, Section 2, of that Regulation shall be taken into account.
(*)

Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised
rules on artificial intelligence and amending Regulations (EC) No 300/2008, (EU) No 167/2013, (EU) No 168/2013,
(EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and Directives 2014/90/EU,

---

[Source: EU_AI_Act.txt, Relevance: 0.689]
n Article 5 of Directive (EU) 2016/943 of the European
Parliament and of the Council (57);
(b) the effective implementation of this Regulation, in particular for the purposes of inspections, investigations or audits;
(c) public and national security interests;
(d) the conduct of criminal or administrative proceedings;
(e) information classified pursuant to Union or national law.
2.
The authorities involved in the application of this Regulation pursuant to paragraph 1 shall request only data that is
strictly

---

[Source: EU_AI_Act.txt, Relevance: 0.685]
used in the Union do not pose unacceptable risks to important Union public interests
as recognised and protected by Union law. On the basis of the New Legislative Framework, as clarified in the
Commission notice ‘The “Blue Guide” on the implementation of EU product rules 2022’ (20), the general rule is that
more than one legal act of Union harmonisation legislation, such as Regulations (EU) 2017/745 (21) and (EU)
2017/746 (22) of the European Parliament and of the Council or Directive 2006/42/EC of the Eur

---

[Source: EU_AI_Act.txt, Relevance: 0.682]
the expiry of that period, the European Parliament and the Council
have both informed the Commission that they will not object. That period shall be extended by three months at the
initiative of the European Parliament or of the Council.

Article 98
Committee procedure
1.
The Commission shall be assisted by a committee. That committee shall be a committee within the meaning of
Regulation (EU) No 182/2011.
2.

Where reference is made to this paragraph, Article 5 of Regulation (EU) No 182/2011 shall apply.

---

[Source: EU_AI_Act.txt, Relevance: 0.681]
/1861 of the European Parliament and of the Council and Council Decisions 2004/512/EC and
2008/633/JHA (OJ L 135, 22.5.2019, p. 27).
(b) Regulation (EU) 2019/818 of the European Parliament and of the Council of 20 May 2019 on establishing
a framework for interoperability between EU information systems in the field of police and judicial cooperation,
asylum and migration and amending Regulations (EU) 2018/1726, (EU) 2018/1862 and (EU) 2019/816 (OJ
L 135, 22.5.2019, p. 85).

140/144
============================================================

Context Scores:
[0.7442622184753418, 0.7286093235015869, 0.714722752571106, 0.6934648156166077, 0.690255343914032, 0.6893728971481323, 0.6891202330589294, 0.6851427555084229, 0.6819180846214294, 0.6806100606918335]
============================================================


```
### Question 6:
```text
QUESTION:
============================================================

How do I adjust the carburetor on a Model T?
============================================================

============================================================
RETRIEVED CONTEXT:
============================================================
[Source: ModelTNew.txt, Relevance: 0.636]
.
  Insert end of rod through throttle lever "B" a nd lock the rod in
  position by inserting a cotter pin through end of rod.

115  Install carburetor adjusting rod by inserting head of rod
  through slot in dash. Place forked end of rod "C" through head of
  carburetor needle valve, locking the rod in position by inserting a
  cotter key through end of rod.
34                         FORD SERVICE




                Fig. 91                          Fig. 92

---

[Source: ModelTNew.txt, Relevance: 0.632]
y withdrawing the two cotter
  pins which hold pull rod to carburetor throttle and throttle rod
  lever (See "A" Fig. 28) .




                  F ig. 29                           Fig. 30

26  Remove carburetor adjusting rod by withdrawing cotter pin
  "B".

---

[Source: ModelTNew.txt, Relevance: 0.617]
CHAPTER XXVII

                Carburetor Overhaul




                 Fig. 426                        Fig . 427



                 Removing Carburetor from Car
867   Lift off hood.
868 Disconnect carburetor pull rod at throttle and adjusting rod at
  needle valve by withdrawing cotter pins (See "A" and " B" Fig. 28).
869 Disconnect the two priming wires at carburetor butterfly (See
  Fig.

---

[Source: ModelTNew.txt, Relevance: 0.614]
f carburetor hot air pipe into carburetor, and
       tighten the nut which holds hot air pipe to manifold (See "A"
       Fig. 426).
  (d) Connect adjusting rod at carburetor needle valve by inserting
       forked end of rod through needle valve and inserting cotter pin
       through end of rod (See "B" Fig. 28).
  (e) Connect the two priming wires at carburetor butterfly (See Fig.
       18)0                                                        •

---

[Source: ModelTNew.txt, Relevance: 0.606]
F ig. 29                           Fig. 30

26  Remove carburetor adjusting rod by withdrawing cotter pin
  "B". Forked end of rod can then be lifted from carburetor needle
  valve and head of rod withdrawn through dash.
27 Remove priming wire by unhooking priming wire "C" at car-
  buretor butterfly and bell crank.
                           FORD SERVICE                              9

---

[Source: ModelTNew.txt, Relevance: 0.595]
d to carburetor by inserting
  forked end of rod into head of needle v alve and locking rod in posi-
  tion with a cotter k ey. (See "C" Fig. 92) .
347 Connect feed pipe to carburetor by running down feed pipe pack
  nut (See " A " Fig. 29).
348 Install fan, fan belt, radiator, horn, and carburetor priming rod
  as described in Pars. 127 to 129.
349 Position hood blocks on frame, making sure that headlamp wire
  bushings enter holes in hook blocks (See " A" Fig. 112) .

---

[Source: ModelTNew.txt, Relevance: 0.593]
tator case by inserting
  end of rod through lever on case, and locking the rod in position with
  a cotter key. (See " B" Fig. 24) the commutator is then checked
  for correct setting as described in Par. 126.
343 Install carburetor pull rod by slipping rod through hole in valve
  cover and inserting ends of rod through carburetor throttle a nd
  throttle rod lever, the rod is then locked in position by inserting
  cotter keys through ends of rod (See " A " Fig.

---

[Source: ModelTNew.txt, Relevance: 0.592]
are then installed over manifold stud "B".
  (c) Place lower end of hot air pipe "C" into carburetor mixing
      chamber "D" . Run down nut on end or' manifold stud "B",
      drawing nut down tightly against manifold clamp.

114  Connect carburetor pull rod to throttle lever b y inserting
  carburetor pull rod through hole in valve door (See "A" Fig. 92) .
  Insert end of rod through throttle lever "B" a nd lock the rod in
  position by inserting a cotter pin through end of rod.

---

[Source: ModelTNew.txt, Relevance: 0.583]
Lift off hood.
169 Disconnect carburetor priming rod at carburetor and withdraw
   rod through r adiator apron (See " A " Fig. 18) .
170   Disconnect bell crank priming wire at carburetor (S ee "C"
   Fig. 28) .
171 Disconnect carburetor adj u sting rod a t carbur etor (See "B"
   Fig. 28) .
172 R emove carburetor pull rod from carburetor and throttle rod
  ·lever (See "A" Fig. 28) .
173 Disconnect feed pipe at carburetor by running off the feed pipe
   pack nut from carburetor elbow (See " A" Fig.

---

[Source: ModelTNew.txt, Relevance: 0.582]
by withdrawing cotter pins (See "A" and " B" Fig. 28).
869 Disconnect the two priming wires at carburetor butterfly (See
  Fig. 18) .
870   Shut off gasoline at sediment bulb underneath gasoline tank .
871 Disconnect feed pipe at carburetor by running off pack nut
  (See Fig. 29 ).
872 Loosen manifold stud nut which holds carburetor hot air pipe
  to manifold (See "A" Fig.
============================================================

Context Scores:
[0.6363170146942139, 0.631974458694458, 0.6174129247665405, 0.6138850450515747, 0.6059490442276001, 0.595123291015625, 0.5930905342102051, 0.591831386089325, 0.5827416777610779, 0.5816732048988342]
============================================================


```
### Question 7:
```text
QUESTION:
============================================================

What is the correct spark plug gap for a Model T Ford?
============================================================

============================================================
RETRIEVED CONTEXT:
============================================================
[Source: ModelTNew.txt, Relevance: 0.564]
0" higher clearance




                Fig. 210                          Fig. 2ll
86                            FORD SERVICE

     t:1an the upper half. Shims (See Fig. 210) are furnished in various
     thicknesses so that extremely close adjustments can be obtained in
     setting the gap.

---

[Source: ModelTNew.txt, Relevance: 0.524]
iddle ring, and .004" to .006" for bottom ring.
427  If the gap is too small, the ends of the ring can be filed (See Fig.
  252), until the correct gap is obtained. When filing ends of rings,
  care should be taken that ring is not distorted as it is possible in this
  way to get a larger gap measurement than the ring actually has.
108                          FORD SERVICE

---

[Source: ModelTNew.txt, Relevance: 0.520]
e floor boards and mat.
1256 Withdra w rubber tubing which was inserted over end of battery
  wire to prevent a spark (see Par. 1238) and insert terminal of wire
  under hea d of battery wire screw on terminal block.
1257 Inst ali hood and fill gasoline tank with fuel.
282                         FORD SERVICE

                 More Room for the Driver




                Fig. 566                           Fig. 567

---

[Source: ModelTNew.txt, Relevance: 0.515]
. . . . . . . . . . . . . . . . . . . . . . . . . . . 149
              installing and removing .... . .. . ....... . ........ 66- 60
Fuel system, tracing trouble in ....... . .. . .. . .. . ... . . . . . . .... 1014
                                                G
Gap, between magnet clamps and coil core. . . . . . . . . . . . . . . . . . . 319
      piston ring . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 426
      spark plug . . . . . . . . . . . . . . . . . . .

---

[Source: ModelTNew.txt, Relevance: 0.503]
and cylinder
   block and insert a small steel pin Ys" x X" driving the pin down tight-
   ly between plug and cylinder block. The plug is then drilled and
   tapped to the original size of the bolt or cap screw hole.
                           FORD SERVICE                               59

            Rebabbitting the Cylinder Block




                Fig. 148                         Fig. 149

---

[Source: ModelTNew.txt, Relevance: 0.498]
oint 10, also the two soldered connections on terminal to bus bar
FORD SERVICE   233
234                        FORD SERVICE

  wire in coil box, and the battery wire, ignition wire and magneto
  wire terminals, points 18, 19 and 20 on back of switch to see that
  they are clean and tight.

---

[Source: ModelTNew.txt, Relevance: 0.490]
ssible in this
  way to get a larger gap measurement than the ring actually has.
108                          FORD SERVICE




428 Rings should next be checked on a surface plate to make sure
  they have not been sprung (See Fig. 253). If alignment of ring is
  0. K. run the ring around the groove in the piston into which it is to be
  fitted (See Fig. 254) . The ring should fit in the groove with a clear-
  ance offrom .001" to .002".

---

[Source: ModelTNew.txt, Relevance: 0.489]
ee Par. 38) and screw in spark plugs, con-
  necting the wires to coil box posts (See " A" Fig. 106) and spark
  plugs "B".




                               F ig. 106

131   Install running boards and shields-
  (a ) Position shield on frame ; insert running board shield to frame;
       bolt through frame and shield, and run down lockwasher and
       nut (See "A" Fig. 17).
  (b ) Place running board blocks (See "A" Fig. 107) on running
       board brackets.
40                         FORD SERVICE

---

[Source: ModelTNew.txt, Relevance: 0.483]
Fig. 255
                                          FORD SERVICE                                                     109

430 After pistons have been checked and rings correctly fitted, turn
  the rings in the groove so that the gaps in the rings will not be in
  line. The gaps should be approximately 120° apart.
431 Before installing pistons in cylinders place oil on sides of pistons
  and wipe out cylinder bores with a cloth free from lint.

---

[Source: ModelTNew.txt, Relevance: 0.481]
2.848




                                                 Fig. 466

  The most convenient method of checking this dimension 1s with a
  "go" and "no go" plug gauge (See Fig. 467).
924 Tighten pole screws with a pole screw driver (See Fig. 462).
  To prevent any possibility of the pole screws working loose, they
  should be staked with a center punch as shown at "A" Fig. 468.
220                         FORD SERVICE
============================================================

Context Scores:
[0.564090371131897, 0.5237500667572021, 0.5203385353088379, 0.5153529644012451, 0.5026880502700806, 0.4983852803707123, 0.48959898948669434, 0.4888741075992584, 0.48291462659835815, 0.480646550655365]
============================================================


```
### Question 8:
```text
QUESTION:
============================================================

How do I fix a slipping transmission band?
============================================================

============================================================
RETRIEVED CONTEXT:
============================================================
[Source: ModelTNew.txt, Relevance: 0.522]
Fig. 30).
522 Insert the two bolts in universal joint. Run down nuts on ends
  ot bolts and lock with cotter key (See "A" Fig. 32 ) .
523 Remove transmission cover door, by running out the screw which
  was used to temporarily hold it in place.
524 Adjust low speed band by running in the adjusting screw (See
  "A" Fig.

---

[Source: ModelTNew.txt, Relevance: 0.507]
run on exhaust pipe nut, put in floor
        boards and mat, and test for oil leaks and adjustment
        on bands . . . . . ...................... .                                     15
                                                                                  1     53
                           CHAPTER XIII
   Installing New Type Transmission
                 Bands




                Fig. 308

---

[Source: ModelTNew.txt, Relevance: 0.496]
ve the lug, simply insert a tool through the end of the
  lug into the square hole in end of brake band, lift up on the tool
  forcing the brake band down and the lug back (See Fig. 309).
535 To remove the new type bands, remove transmission cover door
  (See "B" Fig. 299).

---

[Source: ModelTNew.txt, Relevance: 0.484]
the
  procedure for tightening the clutch as described in Par. 1060. The
  low speed band is adjusted as outlined in Par. 524.

1065  During cold weather, when the oil becomes congealed, it will
  sometimes cause the clutch plates to stick and this will have a
  tendency to make the car creep forward . This condition disappears,
  however, as soon as the engine becomes warm.

---

[Source: ModelTNew.txt, Relevance: 0.481]
rectly below
 the opening in the transmission cover. Withdraw cotter pin from
 clutch finger screw and give the screw (See Fig. 507) one-half turn
 (clockwise), then replace the cotter pin and adjust the two other
 clutch finger screws in the same manner. The transmission cover
 door is then replaced and the clutch tested for correct adjustment.
 If the clutch still slips, give each of the three screws another half
 turn, making sure that all three screws receive exactly the same num-
 ber of half turns.

---

[Source: ModelTNew.txt, Relevance: 0.477]
F ig. 300                        Fig. 301
                             FORD SERVICE                               133




                Fig. 3 0 2                         Fig. 303


 spread and are firmly imbedded in the lining. This prevents any
 possibility of scoring the transmission drums. The bands are in-
 stalled one at a time by slipping them over the triple gears and edge
 of reverse drum, with the lugs downward; while in this position, the
 band is then turned until the lugs are on top.

---

[Source: ModelTNew.txt, Relevance: 0.475]
d.
538 To replace the bands, insert end of band from which lug has been
  removed into right side of the transmission (See Fig. 311), forcing
  the band around until it is possible to reach the square hole with
  the hooked tool shown in Fig. 315. When the end of the band has
                                  136
                           FORD SERVICE                           137




                Fig. 310                        Fig. 311

---

[Source: ModelTNew.txt, Relevance: 0.470]
I

       Replacing Transmission Bands
                             (Old Design)




                                 Fig. 297

501 When transmission band linings become worn to such an extent
  that their braking efficiency is impaired, the bands should be relined.
  (If the car is equipped with the new type bands see Par. 532.

---

[Source: ModelTNew.txt, Relevance: 0.470]
ission cover to change this type band as all of
  the operations can be performed through the transmission cover door.
533 Fig. 308 shows the band with the lug removed. The lug is held
  in place by means of studs in the band which protrude through the
  slots in the lug, the shoulder of the lug snapping over the end of the
  brake band and holding it securely in place.
534 To remove the lug, simply insert a tool through the end of the
  lug into the square hole in end of brake band, lift up on the tool
  f

---

[Source: ModelTNew.txt, Relevance: 0.463]
nd down and the lug back (See Fig. 309).
535 To remove the new type bands, remove transmission cover door
  (See "B" Fig. 299). Run off the nuts and lockwashers from the ends
  of the pedal shafts, run out slow speed adjusting screw and remove
  the springs.
536 "EXTREME CARE SHOULD BE EXERCISED TO PRE-
  VENT DROPPING ANY PARTS INTO THE TRANSMISSION."
537 The reverse pedal and brake pedal should then be pulled out as
  far as they will conveniently go (See Fig.
============================================================

Context Scores:
[0.5217059850692749, 0.5068963766098022, 0.4960765242576599, 0.4841296374797821, 0.48074251413345337, 0.4767594039440155, 0.4747145175933838, 0.46983209252357483, 0.46957892179489136, 0.4631277322769165]
============================================================


```
### Question 9:
```text
QUESTION:
============================================================

What oil should I use in a Model T engine?
============================================================

============================================================
RETRIEVED CONTEXT:
============================================================
[Source: ModelTNew.txt, Relevance: 0.558]
t cover and large
  time gear as described in Pars. 454, 455 and 457.
490 Place nozzle of air hose in end of oil pipe (See Fig. 294) and
  blow out any foreign matter which may have accumulated in pipe.
  If no compressed air is available the feed pipe can be cleaned by
  forcing a small flexible cable through it.
491 Turn motor over rapidly with starting crank to see that oil
  flows in an even stream out end of tube.
492        Drain the old oil from crankcase and pour in a gallon of new
    oil.
493 Repl

---

[Source: ModelTNew.txt, Relevance: 0.488]
ows in an even stream out end of tube.
492        Drain the old oil from crankcase and pour in a gallon of new
    oil.
493 Replace time gear as described in "i" Par. 458. Replace cylinder
  front cover gasket and cover and commutator as c\escribed in .Pars.
  466 to 469. Install fan and radiator as described in Pars. 127 and
  128. Install hood ; close drain cock at bottom of ra diator and fill
  radiator with clean water .

---

[Source: ModelTNew.txt, Relevance: 0.424]
. . . . . . 122
                                 CHAPTER X
Cleaning the oil line. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 128
                             CHAPTER X I
Stopping oil leak a t front end of crankshaft. . . . . . . . . . . . . . . . . .                           129
                           CHAPTER X I 1
Installing transmission bands (old design). . . . . . . . . . . . . . . . . . . .

---

[Source: ModelTNew.txt, Relevance: 0.400]
CHAPTER X

                    Cleaning the Oil Line




                                       Fig. 294

489 Lift off hood and remove radiator and fan as described in Pars.
  14 and 16. Remove commutator, cylinder front cover and large
  time gear as described in Pars. 454, 455 and 457.
490 Place nozzle of air hose in end of oil pipe (See Fig.

---

[Source: ModelTNew.txt, Relevance: 0.388]
time gears . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44 7
           valves.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1046

                                                         0
     Oil leak at front end of crankcase . . . . . . . . . . . . . . . . . . . . . . . . . . . 495
                                     crankshaft. ....... ... . .. . .. .... . . . ... 495
                 rear wheel. . . . . . . .

---

[Source: ModelTNew.txt, Relevance: 0.387]
.. . ..... . . . .               28
4     Install hood, fill radiator with water, remove car covers .. .                     8

                                                                                   1    18
                                          128
                            CHAPTER XI

      Stopping Oil Leak at Front End
               of Crankshaft




                 Fig. 295                          Fig. 296

---

[Source: ModelTNew.txt, Relevance: 0.377]
run on exhaust pipe nut, put in floor
        boards and mat, and test for oil leaks and adjustment
        on bands . . . . . ...................... .                                     15
                                                                                  1     53
                           CHAPTER XIII
   Installing New Type Transmission
                 Bands




                Fig. 308

---

[Source: ModelTNew.txt, Relevance: 0.375]
hed onto cylinder block, making
  sure that the gaskets on the exhaust arid inta ke pipes fit tightly
  against cylinder block. Place the four inlet and exhaust pipe clamps
  "A" over ends of studs and run down the four manifold stud nuts.
333 Pour one gallon of a light high grade engine oil into crankcase
  through breather pipe.
334 The complete assembly is now placed on a running-in machine,
  and run in for a period of at least 20 minutes and carefully checked
  for oil lea ks, and knocks which might po

---

[Source: ModelTNew.txt, Relevance: 0.373]
. . . . . . . . . . . . . . . . . . . . . . . . . 495
Crankshaft, end play in.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 992
              fitting and running in . . . . . . . . . . . . . . . . . . . . . . . . . . 244
              oil leak at front end . . . . . . . . . . . . . . . . . . . . . . . . . . . . 495
Cup, front hub bearing, removing and installing. . . . . . . . . . . . . . 731
Cylinder block, inspection of. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

---

[Source: ModelTNew.txt, Relevance: 0.373]
r ot ation until both v a lve and seat have been ground.
       T o facilitat e lifting v alve when grinding, insert a spring over
       end of v alve (a starting crank spring can be used for this pur-
       pose) (See "A" Fig. 22 3) . The valve should not be turned through
       a complete revolution a t one time, as this is apt to cause scratches
       runnin g a round the entire circumference of the valve and seat .
371   The id eal sea t for valves in an internal combustion engine is a
       h a i
============================================================

Context Scores:
[0.557917058467865, 0.4880726933479309, 0.4241105914115906, 0.40031296014785767, 0.38827747106552124, 0.3869420289993286, 0.37736445665359497, 0.37526512145996094, 0.3730238676071167, 0.3729400038719177]
============================================================


```
### Question 10:
```text
QUESTION:
============================================================

How should the notion of ‘biometric categorisation’, referred to in Regulation 2024/1689, be defined?
============================================================

============================================================
RETRIEVED CONTEXT:
============================================================
[Source: EU_AI_Act.txt, Relevance: 0.868]
C (Digital Services Act) (OJ L 277, 27.10.2022, p. 1).

ELI: http://data.europa.eu/eli/reg/2024/1689/oj

OJ L, 12.7.2024
(16)

The notion of ‘biometric categorisation’ referred to in this Regulation should be defined as assigning natural persons
to specific categories on the basis of their biometric data.

---

[Source: EU_AI_Act.txt, Relevance: 0.742]
ith the applicable law should not, in themselves, be regarded as constituting harmful
manipulative AI-enabled practices.

(30)

Biometric categorisation systems that are based on natural persons’ biometric data, such as an individual person’s
face or fingerprint, to deduce or infer an individuals’ political opinions, trade union membership, religious or
philosophical beliefs, race, sex life or sexual orientation should be prohibited.

---

[Source: EU_AI_Act.txt, Relevance: 0.670]
be put in place or into the market for medical or safety reasons;
ELI: http://data.europa.eu/eli/reg/2024/1689/oj

51/144

EN

OJ L, 12.7.2024
(g) the placing on the market, the putting into service for this specific purpose, or the use of biometric categorisation
systems that categorise individually natural persons based on their biometric data to deduce or infer their race, political
opinions, trade union membership, religious or philosophical beliefs, sex life or sexual orientation; this prohibition doe

---

[Source: EU_AI_Act.txt, Relevance: 0.669]
in this Regulation should be defined as assigning natural persons
to specific categories on the basis of their biometric data. Such specific categories can relate to aspects such as sex,
age, hair colour, eye colour, tattoos, behavioural or personality traits, language, religion, membership of a national
minority, sexual or political orientation.

---

[Source: EU_AI_Act.txt, Relevance: 0.656]
of law enforcement as regulated by this Regulation, should continue to comply
with all requirements resulting from Article 10 of Directive (EU) 2016/680. For purposes other than law
enforcement, Article 9(1) of Regulation (EU) 2016/679 and Article 10(1) of Regulation (EU) 2018/1725 prohibit the
processing of biometric data subject to limited exceptions as provided in those Articles.

---

[Source: EU_AI_Act.txt, Relevance: 0.649]
t (14) of Regulation (EU) 2016/679, Article 3, point (18) of Regulation (EU) 2018/1725
and Article 3, point (13) of Directive (EU) 2016/680. Biometric data can allow for the authentication, identification
or categorisation of natural persons and for the recognition of emotions of natural persons.

(15)

---

[Source: EU_AI_Act.txt, Relevance: 0.643]
es other than law
enforcement has already been subject to prohibition decisions by national data protection authorities.

(40)

In accordance with Article 6a of Protocol No 21 on the position of the United Kingdom and Ireland in respect of the
area of freedom, security and justice, as annexed to the TEU and to the TFEU, Ireland is not bound by the rules laid
down in Article 5(1), first subparagraph, point (g), to the extent it applies to the use of biometric categorisation
systems for activities in the fiel

---

[Source: EU_AI_Act.txt, Relevance: 0.635]
tion under this Regulation and the applicable detailed rules of national law
that may give effect to that authorisation.

(39)

Any processing of biometric data and other personal data involved in the use of AI systems for biometric
identification, other than in connection to the use of real-time remote biometric identification systems in publicly
accessible spaces for the purpose of law enforcement as regulated by this Regulation, should continue to comply
with all requirements resulting from Article 10 of

---

[Source: EU_AI_Act.txt, Relevance: 0.631]
egulation (EU) 2018/1725 prohibit the
processing of biometric data subject to limited exceptions as provided in those Articles. In the application of Article
9(1) of Regulation (EU) 2016/679, the use of remote biometric identification for purposes other than law
enforcement has already been subject to prohibition decisions by national data protection authorities.

(40)

---

[Source: EU_AI_Act.txt, Relevance: 0.630]
tical
opinions, trade union membership, religious or philosophical beliefs, sex life or sexual orientation; this prohibition does
not cover any labelling or filtering of lawfully acquired biometric datasets, such as images, based on biometric data or
categorizing of biometric data in the area of law enforcement;
(h) the use of ‘real-time’ remote biometric identification systems in publicly accessible spaces for the purposes of law
enforcement, unless and in so far as such use is strictly necessary for one o
============================================================

Context Scores:
[0.8683453798294067, 0.7415913939476013, 0.669829249382019, 0.6691179275512695, 0.6556612253189087, 0.648659884929657, 0.643451452255249, 0.6353139877319336, 0.6305447220802307, 0.6304410099983215]
============================================================


```
### Threshold Demo:
```text
QUESTION:
============================================================

What oil should I use in a Model T engine?
============================================================

============================================================
RETRIEVED CONTEXT:
============================================================
[Source: ModelTNew.txt, Relevance: 0.558]
t cover and large
  time gear as described in Pars. 454, 455 and 457.
490 Place nozzle of air hose in end of oil pipe (See Fig. 294) and
  blow out any foreign matter which may have accumulated in pipe.
  If no compressed air is available the feed pipe can be cleaned by
  forcing a small flexible cable through it.
491 Turn motor over rapidly with starting crank to see that oil
  flows in an even stream out end of tube.
492        Drain the old oil from crankcase and pour in a gallon of new
    oil.
493 Repl

---

[Source: ModelTNew.txt, Relevance: 0.488]
ows in an even stream out end of tube.
492        Drain the old oil from crankcase and pour in a gallon of new
    oil.
493 Replace time gear as described in "i" Par. 458. Replace cylinder
  front cover gasket and cover and commutator as c\escribed in .Pars.
  466 to 469. Install fan and radiator as described in Pars. 127 and
  128. Install hood ; close drain cock at bottom of ra diator and fill
  radiator with clean water .

---

[Source: ModelTNew.txt, Relevance: 0.424]
. . . . . . 122
                                 CHAPTER X
Cleaning the oil line. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 128
                             CHAPTER X I
Stopping oil leak a t front end of crankshaft. . . . . . . . . . . . . . . . . .                           129
                           CHAPTER X I 1
Installing transmission bands (old design). . . . . . . . . . . . . . . . . . . .

---

[Source: ModelTNew.txt, Relevance: 0.400]
CHAPTER X

                    Cleaning the Oil Line




                                       Fig. 294

489 Lift off hood and remove radiator and fan as described in Pars.
  14 and 16. Remove commutator, cylinder front cover and large
  time gear as described in Pars. 454, 455 and 457.
490 Place nozzle of air hose in end of oil pipe (See Fig.

---

[Source: ModelTNew.txt, Relevance: 0.388]
time gears . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44 7
           valves.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1046

                                                         0
     Oil leak at front end of crankcase . . . . . . . . . . . . . . . . . . . . . . . . . . . 495
                                     crankshaft. ....... ... . .. . .. .... . . . ... 495
                 rear wheel. . . . . . . .

---

[Source: ModelTNew.txt, Relevance: 0.387]
.. . ..... . . . .               28
4     Install hood, fill radiator with water, remove car covers .. .                     8

                                                                                   1    18
                                          128
                            CHAPTER XI

      Stopping Oil Leak at Front End
               of Crankshaft




                 Fig. 295                          Fig. 296

---

[Source: ModelTNew.txt, Relevance: 0.377]
run on exhaust pipe nut, put in floor
        boards and mat, and test for oil leaks and adjustment
        on bands . . . . . ...................... .                                     15
                                                                                  1     53
                           CHAPTER XIII
   Installing New Type Transmission
                 Bands




                Fig. 308

---

[Source: ModelTNew.txt, Relevance: 0.375]
hed onto cylinder block, making
  sure that the gaskets on the exhaust arid inta ke pipes fit tightly
  against cylinder block. Place the four inlet and exhaust pipe clamps
  "A" over ends of studs and run down the four manifold stud nuts.
333 Pour one gallon of a light high grade engine oil into crankcase
  through breather pipe.
334 The complete assembly is now placed on a running-in machine,
  and run in for a period of at least 20 minutes and carefully checked
  for oil lea ks, and knocks which might po

---

[Source: ModelTNew.txt, Relevance: 0.373]
. . . . . . . . . . . . . . . . . . . . . . . . . 495
Crankshaft, end play in.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 992
              fitting and running in . . . . . . . . . . . . . . . . . . . . . . . . . . 244
              oil leak at front end . . . . . . . . . . . . . . . . . . . . . . . . . . . . 495
Cup, front hub bearing, removing and installing. . . . . . . . . . . . . . 731
Cylinder block, inspection of. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

---

[Source: ModelTNew.txt, Relevance: 0.373]
r ot ation until both v a lve and seat have been ground.
       T o facilitat e lifting v alve when grinding, insert a spring over
       end of v alve (a starting crank spring can be used for this pur-
       pose) (See "A" Fig. 22 3) . The valve should not be turned through
       a complete revolution a t one time, as this is apt to cause scratches
       runnin g a round the entire circumference of the valve and seat .
371   The id eal sea t for valves in an internal combustion engine is a
       h a i
============================================================

Context Scores:
[0.557917058467865, 0.4880726933479309, 0.4241105914115906, 0.40031296014785767, 0.38827747106552124, 0.3869420289993286, 0.37736445665359497, 0.37526512145996094, 0.3730238676071167, 0.3729400038719177]
============================================================

============================================================
RETRIEVED CONTEXT (RELEVANCE >= 0.5):
============================================================
[Source: ModelTNew.txt, Relevance: 0.558]
t cover and large
  time gear as described in Pars. 454, 455 and 457.
490 Place nozzle of air hose in end of oil pipe (See Fig. 294) and
  blow out any foreign matter which may have accumulated in pipe.
  If no compressed air is available the feed pipe can be cleaned by
  forcing a small flexible cable through it.
491 Turn motor over rapidly with starting crank to see that oil
  flows in an even stream out end of tube.
492        Drain the old oil from crankcase and pour in a gallon of new
    oil.
493 Repl
============================================================


```