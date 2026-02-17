# Exercise 7: Chunk Overlap Experiment
Test how overlap between chunks affects retrieval of information that spans chunk boundaries. You can use any of the copora and your own queries. Note: this exercise takes a long time to run.  Only try it on CoLab or a similar platform with T4 or better GPUs.

**Setup:** Re-chunk your corpus with different overlap values while keeping chunk size constant (e.g., 512 characters):

- Overlap = 0 (no overlap)

- Overlap = 64

- Overlap = 128

- Overlap = 256

**For each configuration:**

- Rebuild the index

- Find a question whose answer spans what would be a chunk boundary

- Test retrieval quality

**Document:**

- Does higher overlap improve retrieval of complete information?

- What's the cost? (Index size, redundant information in context)

- Is there a point of diminishing returns?

## Answers
### Does higher overlap improve retrieval of complete information?
Surprising not really. For all the questions I could come up with (that landed on a chunk bound), both relevant chunks would pop up. So with 0 overlap it basically always got the needed info. The overlap increased redundant info but also made it less likely that the full needed context would get found since more overlapping chunks were found. See example of 0 over lap bellow:
``` text
 Answer this question:
For the Quality management system, what should the application of the provider include?

Answer:
WITH CHUNK SIZE 512 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 42/42 [00:01<00:00, 26.93it/s]Rebuilt: 1339 chunks, chunk_size=512, chunk_overlap=0
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

The application of the provider for the Quality management system should include:

(a) the name and address of the provider and, if the application is lodged by an authorised representative, also their name and address;

(b) the list of AI systems covered under the same quality management system;

(c) the technical documentation for each AI system covered under the same quality management system;

(d) the documentation concerning the quality management system which shall cover all the aspects listed under Article 17;

(e) a description of the procedures involved in the quality management system; 

(f) a written declaration that the same application has not been lodged with any other notified body. 

Additionally, providers of high-risk AI systems must ensure that the quality management system includes the elements specified in the Regulation, and any modifications to the system or the list of AI systems must be reported to the notified body for examination. To maintain the adequacy and effectiveness of the quality management system, the provider must also undergo regular assessments by the notified body.

```

### What's the cost? (Index size, redundant information in context)
More redundant info but also less space for the full needed context because more overlap is found.

### Is there a point of diminishing returns?
yes: 0 overlap was best i think. I assume that there must be some benefit to overlap but did not find evidence of that with my questions.

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
Missing Text
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

## (a) Regulation (EU) 2021/1133

This regulation:

* Was adopted on **7 July 2021** by the European Parliament and the Council.
* **Amends several existing EU regulations** (such as 603/2013, 2016/794, 2018/1862, 2019/816, 2019/818).
* Its main goal is to **set the legal conditions under which the Visa Information System (VIS) can access other EU information systems.**

### In simple terms:

It creates the legal framework that allows the **Visa Information System (VIS)** to connect with and access data from other EU databases — but only under defined rules and conditions.

So this regulation is about **interoperability and access rules** between EU information systems.

---

## (b) Regulation (EU) 2021/1134

This regulation:

* Was also adopted on **7 July 2021**.
* Amends a large number of EU regulations related to borders, visas, migration, and security.
* **Repeals two older Council Decisions.**
* Its purpose is explicitly stated as **reforming the Visa Information System.**

### In simple terms:

This regulation **updates and modernizes the Visa Information System itself.**

It changes the core legal framework governing how VIS operates, likely expanding its functions, improving data use, and aligning it with newer EU border and security systems.

---

## The Key Difference

* **(a) = How VIS connects to other EU systems (access conditions).**
* **(b) = Reforming and upgrading the VIS system itself.**

So:

* (a) focuses on **cross-system access rules**.
* (b) focuses on **structural reform of VIS and related legislation**.

If you're testing RAG chunk boundaries, the conceptual distinction between interoperability (a) and reform (b) could also be useful for crafting edge-spanning questions.

## Question on Edge
### Q1
```
Under “Visa Information System”, when was Regulation (EU) 2021/1133 adopted?
```
good because date and reg are seperated.

### Q2
```
Under “Visa Information System”, what is the Regulation number listed in point (b), and what is its stated purpose?
```
good because seperate chunks.

### Q3
```text
Under “Visa Information System”, which Regulation establishes conditions for accessing other EU information systems, and which Regulation reforms the Visa Information System?
```


## Response:
```
 Answer this question:
Under “Visa Information System”, when was Regulation (EU) 2021/1133 adopted?

Answer:
WITH CHUNK SIZE 512 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 82/82 [00:03<00:00, 30.54it/s]Rebuilt: 2601 chunks, chunk_size=512, chunk_overlap=0
Regulation (EU) 2021/1133 was adopted on 7 July 2021. This regulation amended several existing regulations related to the Visa Information System, including Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816, and (EU) 2019/818. The adoption date is explicitly stated in the first source document provided.
WITH CHUNK SIZE 512 and CHUNK OVERLAP 64 (using retrieved context):
----------------------------------------
Batches: 100% 96/96 [00:04<00:00, 27.67it/s]Rebuilt: 3045 chunks, chunk_size=512, chunk_overlap=64
Regulation (EU) 2021/1133 was adopted on 7 July 2021. This regulation is mentioned under "Visa Information System" in the context, specifically in the first paragraph where it states "(b) Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021 amending Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA". The adoption date is clearly stated as 7 July 2021.
WITH CHUNK SIZE 512 and CHUNK OVERLAP 128 (using retrieved context):
----------------------------------------
Batches: 100% 114/114 [00:06<00:00, 27.37it/s]Rebuilt: 3632 chunks, chunk_size=512, chunk_overlap=128
Regulation (EU) 2021/1133 was adopted on 7 July 2021. This regulation amended Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards the establishment of the conditions for accessing other EU information systems, specifically those related to the Visa Information System. The adoption date is mentioned directly in the text under "Visa Information System".
WITH CHUNK SIZE 512 and CHUNK OVERLAP 256 (using retrieved context):
----------------------------------------
Batches: 100% 180/180 [00:09<00:00, 28.20it/s]Rebuilt: 5748 chunks, chunk_size=512, chunk_overlap=256
Regulation (EU) 2021/1133 was adopted on 7 July 2021. This regulation amended Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards the establishment of the conditions for accessing other EU information systems for the purposes of the Visa Information System. The adoption date is explicitly mentioned in the first source text.

```

```
 Answer this question:
Under “Visa Information System”, what is the Regulation number listed in point (b), and what is its stated purpose?

Answer:
WITH CHUNK SIZE 512 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 82/82 [00:05<00:00, 28.51it/s]Rebuilt: 2601 chunks, chunk_size=512, chunk_overlap=0
The Regulation number listed in point (b) under "Visa Information System" is (EU) 2021/1134. Its stated purpose is to amend Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the Visa Information System. This regulation aims to update and improve the system by addressing various aspects such as visa application processes, data sharing with member states, and security measures. It also includes provisions related to the ETIAS, which is another aspect of the Visa Information System. The main goal is to enhance the efficiency and effectiveness of the Visa Information System while ensuring compliance with EU regulations and maintaining the integrity of the system.
WITH CHUNK SIZE 512 and CHUNK OVERLAP 64 (using retrieved context):
----------------------------------------
Batches: 100% 96/96 [00:04<00:00, 27.07it/s]Rebuilt: 3045 chunks, chunk_size=512, chunk_overlap=64
The Regulation number listed in point (b) under "Visa Information System" is (EU) 2016/399. Its stated purpose is to amend Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1860, (EU) 2018/1861, and (EU) 2019/817 of the European Parliament and of the Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the Visa Information System.
WITH CHUNK SIZE 512 and CHUNK OVERLAP 128 (using retrieved context):
----------------------------------------
Batches: 100% 114/114 [00:06<00:00, 25.93it/s]Rebuilt: 3632 chunks, chunk_size=512, chunk_overlap=128
The Regulation number listed in point (b) under "Visa Information System" is (EU) 2016/399. Its stated purpose is to amend Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the Council and repealing Council Decision 2007/533/JHA, for the purpose of reforming the Visa Information System.
WITH CHUNK SIZE 512 and CHUNK OVERLAP 256 (using retrieved context):
----------------------------------------
Batches: 100% 180/180 [00:09<00:00, 28.43it/s]Rebuilt: 5748 chunks, chunk_size=512, chunk_overlap=256
The Regulation number listed in point (b) is (EU) 2021/1134, which amends Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the Council and repeals Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the Visa Information System. Its stated purpose is to amend Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards the establishment of the conditions for accessing other EU information systems for the purposes of the Visa Information System.

```

```
 Answer this question:
Under Visa Information System, which Regulation establishes conditions for accessing other EU information systems, and which Regulation reforms the Visa Information System?

Answer:
WITH CHUNK SIZE 512 and CHUNK OVERLAP 0 (using retrieved context):
----------------------------------------
Batches: 100% 82/82 [00:03<00:00, 29.85it/s]Rebuilt: 2601 chunks, chunk_size=512, chunk_overlap=0
The Regulation that establishes conditions for accessing other EU information systems under the Visa Information System is Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021. This regulation amends several existing regulations related to the Visa Information System. The regulation that reforms the Visa Information System is Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021. This regulation amends several existing regulations related to the Visa Information System.
WITH CHUNK SIZE 512 and CHUNK OVERLAP 64 (using retrieved context):
----------------------------------------
Batches: 100% 96/96 [00:04<00:00, 30.88it/s]Rebuilt: 3045 chunks, chunk_size=512, chunk_overlap=64
The Visa Information System is regulated by two main pieces of legislation:

1. **Regulation (EU) 2021/1133** of the European Parliament and of the Council of 7 July 2021, which amends several existing regulations concerning the conditions for accessing other EU information systems for the purposes of the Visa Information System.
   
2. **Regulation (EU) 2021/1134** of the European Parliament and of the Council of 7 July 2021, which amends certain regulations related to the Visa Information System with the aim of reforming it. 

These two regulations together govern the access conditions and the reform process for the Visa Information System under the Visa Information System.
WITH CHUNK SIZE 512 and CHUNK OVERLAP 128 (using retrieved context):
----------------------------------------
Batches: 100% 114/114 [00:05<00:00, 27.29it/s]Rebuilt: 3632 chunks, chunk_size=512, chunk_overlap=128
The Regulation establishing conditions for accessing other EU information systems under the Visa Information System is Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021, which amends Regulations (EU) No 603/2013, (EU) 2016/794, (EU) 2018/1862, (EU) 2019/816 and (EU) 2019/818 as regards the establishment of the conditions for accessing other EU information systems for the purposes of the Visa Information System. The Regulation reforming the Visa Information System is Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021, which amends Regulations (EC) No 767/2008, (EC) No 810/2009, (EU) 2016/399, (EU) 2017/2226, (EU) 2018/1240, (EU) 2018/1860, (EU) 2018/1861, (EU) 2019/817 and (EU) 2019/1896 of the European Parliament and of the Council and repealing Council Decisions 2004/512/EC and 2008/633/JHA, for the purpose of reforming the Visa Information System. To summarize, Regulation (EU) 2021/1133 amends the existing regulations concerning access to EU information systems related to the Visa Information System, while Regulation (EU) 2021/1134 focuses on the overall reform of the system itself.
WITH CHUNK SIZE 512 and CHUNK OVERLAP 256 (using retrieved context):
----------------------------------------
Batches: 100% 180/180 [00:10<00:00, 23.74it/s]Rebuilt: 5748 chunks, chunk_size=512, chunk_overlap=256
The Regulations that establish conditions for accessing other EU information systems for the purposes of the Visa Information System are:

1. Regulation (EU) 2021/1133 of the European Parliament and of the Council of 7 July 2021.
2. Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021.

The Regulation that reforms the Visa Information System is:

1. Regulation (EU) 2021/1134 of the European Parliament and of the Council of 7 July 2021. 

These regulations address different aspects of the Visa Information System, with one focusing on setting conditions for access and another on reforming it.

```