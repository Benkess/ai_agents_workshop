# Question
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