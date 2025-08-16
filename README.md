### **Context Engineering**

- Agents need context (e.g., instructions, external knowledge, tool feedback) to perform tasks

> ***Context engineering is the art and science of filling the context window with just the right information at each step of an agent’s trajectory***
> 

### Analogy

- LLMs are a new kind of OS
    - LLM is CPU
    - Context window is RAM or “working memory” and has [limited capacity](https://lilianweng.github.io/posts/2023-06-23-agent/) to handle context
    - Curation of what fits into RAM is analogous to “context engineering” as mentioned above



### Types of context

[Umbrella discipline](https://x.com/dexhorthy/status/1933283008863482067) that captures a few different types of context:

- **Instructions** – prompts, memories, few‑shot examples, tool descriptions, etc
- **Knowledge** – facts, memories, etc
- **Tools** – feedback from tool calls
 <img width="1672" height="974" alt="image" src="https://github.com/user-attachments/assets/4519ce06-53db-41c8-809e-d5984112f0e7" />

### Why this is harder for agents

- Long-running tasks and accumulating feedback from tool calls
- Agents often utilize a large number of tokens!

<img width="1415" height="581" alt="image" src="https://github.com/user-attachments/assets/27ac8348-c312-4f67-b757-a8e594cd7b48" />

# Problems we might face in context understanding

- [Context Poisoning: When a hallucination makes it into the context](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html#context-poisoning)

**Example:**

- Suppose the assistant once misreads a user’s portfolio and records that the user owns **500 shares of Tesla**, when in reality they own 50.
- Later, when giving advice like “You might consider selling some Tesla shares,” it bases the suggestion on the incorrect 500-share figure.

- [Context Distraction: When the context overwhelms the training](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html#context-distraction)

User asks: *“What’s my current portfolio value?”*

Assistant loads **all past chats, investment history, stock news, SIP reminders, taxes, and debt info**. It gets “distracted” by all this extra context and gives a long, confusing answer about unrelated investment options.

- [Context Confusion: When superfluous context influences the response](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html#context-confusion)

User asks: *“Should I invest in mutual fund X?”*

The assistant remembers the user once said, “I like high-risk crypto investments.” It unnecessarily **biases the advice towards riskier options**, even though the user’s question is about a conservative mutual fund. Recommendations become **misaligned with the user’s current intent or risk profile**.

- [Context Clash: When parts of the context disagree](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html#context-clash)
- Assistant has stored:
    - “User has $10,000 in a savings account”
    - “User has $15,000 in a savings account”
- When asked: *“How much money do I have?”*, the assistant is **unsure which is correct**.
- Context engineering is critical when building agents!

| Failure Type | What Happens | Example | Risk | Solution |
| --- | --- | --- | --- | --- |
| Context Poisoning | False info gets stored | Wrong portfolio shares recorded | Bad advice | Validate inputs, confirm before storing |
| Context Distraction | Too much info overwhelms | Loads all history for simple query | Slow or off-topic answers | Prioritize relevant context, prune memory |
| Context Confusion | Irrelevant info biases answers | Past crypto preference influences mutual fund advice | Misaligned recommendations | Filter context by relevance, tag categories |
| Context Clash | Conflicting info stored | Different balance entries | Inaccurate answers | Timestamp, use authoritative sources, confirm with user |

> *Context Engineering is effectively the #1 job of engineers building AI agents.*
> 

### Approaches

- *Writing context means saving it outside the context window to help an agent perform a task.*
- *Selecting context means pulling it into the context window to help an agent perform a task.*
- *Compressing context involves retaining only the tokens required to perform a task.*
- *Isolating context involves splitting it up to help an agent perform a task.*

<img width="2048" height="729" alt="image" src="https://github.com/user-attachments/assets/47d6d749-5e1e-4144-ac36-fed72800d1bd" />


### 1. Write Context

- *Writing context means saving it outside the context window to help an agent perform a task.*
- Every document is directly embedded and stored in FAISS.
- When a query is asked, FAISS retrieves the nearest documents from the entire dataset.
- There is no compression or separation – it’s a full index.

**Code behavior**

```python
# Index creation
write_index = faiss.IndexFlatL2(embedding_size)
write_index.add(all_embeddings)

# Retrieval
D, I = write_index.search(query_embedding, top_k)
retrieved_docs = [all_docs[i] for i in I[0]]
```

**Use cases/Examples**

- User asks for an **exact transaction**:
    
    “How much rent did I pay on day 2030?”
    
- Best when you want **detailed, raw entries**.

> *The LeadResearcher begins by thinking through the approach and saving its plan to Memory to persist the context, since if the context window exceeds 200,000 tokens it will be truncated and it is important to retain the plan.*
> 

**Memories**

- [Generative Agents](https://ar5iv.labs.arxiv.org/html/2304.03442) synthesized memory from collections of past agent feedback
- [ChatGPT](https://help.openai.com/en/articles/8590148-memory-faq), [Cursor](https://forum.cursor.com/t/0-51-memories-feature/98509), and [Windsurf](https://docs.windsurf.com/windsurf/cascade/memories) all auto-generate memories

<img width="1504" height="920" alt="image" src="https://github.com/user-attachments/assets/9dd18189-d896-4e7c-9656-3c5d232e3ec7" />


### 2. Select Context

- *Selecting context means pulling it into the context window to help an agent perform a task.*
- Instead of searching across the full dataset, documents are first **filtered** by rules or metadata (e.g., only “rent” transactions).
- After filtering, only the reduced set is embedded and searched with FAISS.

**Code behavior**

```python
# Example filtering before FAISS
filtered_docs = [doc for doc in all_docs if "rent" in doc.lower()]
filtered_embeddings = model.encode(filtered_docs)

select_index = faiss.IndexFlatL2(embedding_size)
select_index.add(filtered_embeddings)

# Retrieval
D, I = select_index.search(query_embedding, top_k)
retrieved_docs = [filtered_docs[i] for i in I[0]]
```

**Use cases**

- Queries where you know the **domain keyword in advance**.
    
    Example: “How much rent do I pay?” → filter by “rent” first.
    
- Useful in **very small datasets**, but filtering becomes slow at scale.

### 3. Compress Context

- *Compressing context involves retaining only the tokens required to perform a task.*

**Main idea: Summarization**

**Idea:**

Instead of storing raw logs, store compressed or summarized documents (e.g., *"Total rent in August = ₹32,786"*). Queries then access only this compressed knowledge.

**Pros:**

- Super fast retrieval
- Storage efficient
- Scales well to very large datasets

**Cons:**

- Loses detail (can't answer specific questions like *"What was rent on 12th May?"*)
- Lower accuracy for fact-level queries

**Best Use Case:**

Aggregations, trends, and summaries (*"What's my average savings?"*, *"How much did I spend on rent in August?"*)

<img width="1406" height="1042" alt="image" src="https://github.com/user-attachments/assets/06f9ed35-5bee-4621-aa9f-6f50c64f9629" />


### 4. Isolate Context

- *Isolating context involves splitting it up to help an agent perform a task.*

**Main idea: Multi-agent**

**Idea:**

Split documents into **domain-specific FAISS indexes** (e.g., `budget`, `investment`, `savings`). Query only the relevant domain index.

**Pros:**

- Fast retrieval.
- Scales well with multiple domains.
- More accurate than compress when exact values matter.

**Cons:**

- Query spanning multiple domains can miss context.
- Needs careful schema design (deciding domains).

**Best Use Case:**

Day-to-day Q&A where queries clearly belong to one domain (*“What is my savings balance?”*, *“Show my investments”*).

<img width="1262" height="686" alt="image" src="https://github.com/user-attachments/assets/4df5824b-325c-45d8-aa09-ed04d5e418ec" />


### Results

# Retrieval Approaches – Results Comparison

We ran evaluations across **75,000 financial documents** with the four retrieval strategies.

---

### Write Context

```json
{'accuracy': 1.0, 'size': 75000, 'query_time': 0.0709}
```

- **Accuracy**: 100% (always finds correct entries, since everything is stored).
- **Index Size**: 75,000 vectors (full dataset).
- **Query Time**: ~0.07s per query.
- **Interpretation**: Reliable, but scaling cost is high as all raw transactions are embedded.

---

### Select Context

```json
{'accuracy': 1.0, 'size': 75000, 'query_time': 0.0522}
```

- **Accuracy**: 100% (after filtering + retrieval).
- **Index Size**: 75,000 (same as Write, but filtered dynamically at query).
- **Query Time**: ~0.05s (slightly faster due to narrowed search).
- **Interpretation**: Good if queries can be filtered (e.g., keyword “rent”), but inefficient if filters require scanning large text.

---

### Compress Context

```json
{'accuracy': 0.67, 'size': 7, 'query_time': 0.0262}
```

- **Accuracy**: ~67% (only correct for high-level queries, not raw details).
- **Index Size**: 7 vectors (summaries instead of raw docs).
- **Query Time**: ~0.026s (fastest, since only 7 items to search).
- **Interpretation**: Excellent for **reports/aggregates**, but loses precision for detailed queries.

---

### Isolate Context

```json
{'accuracy': 1.0, 'size': 75000, 'query_time': 0.0414}
```

- **Accuracy**: 100% (domain separation reduces ambiguity).
- **Index Size**: 75,000 (same as Write, but partitioned).
- **Query Time**: ~0.041s (fastest among full-size indexes).
- **Interpretation**: Best tradeoff — keeps raw details, but routing to domain-specific FAISS indexes makes queries faster.

---

### What is Query Time?(Just for reference)

- **Query Time** is the **time taken to embed the query**.
- It reflects the **latency** the chatbot will experience when answering a user’s question.
- Lower is better, especially when dataset size grows.

---

### Takeaway from this experiment

- **Write Context**: Reliable baseline, but not scalable.
- **Select Context**: Works only if queries can be pre-filtered.
- **Compress Context**: Great for dashboards/summary queries, not precise for detail.
- **Isolate Context**: Best overall balance for **75k dataset** — accurate, detailed, and relatively fast.

| Approach | Accuracy | Index Size | Query Time (s) | Explanation |
| --- | --- | --- | --- | --- |
| **Write Context** | 1.0 | 75,000 | 0.0709 | Stores all raw docs. Accurate but heavy. Scales poorly. |
| **Select Context** | 1.0 | 75,000 | 0.0522 | Same as Write, but filters on query. Slightly faster. |
| **Compress Context** | 0.67 | 7 | 0.0262 | Stores only summaries. Very fast, but loses detail accuracy. |
| **Isolate Context** | 1.0 | 75,000 | 0.0414 | Splits data into domains. Accurate, scalable, and faster than Write. |

### Query based Experiment

### Query 1: *“How much rent do I pay?”*

| Approach | Query Time (s) | Retrieved Results |
| --- | --- | --- |
| **Write Context** | 0.0276 | Paid ₹2500 (day 44060), Paid ₹395 (day 10020) |
| **Select Context** | 21.9548 | Paid ₹2500 (day 44060), Paid ₹395 (day 10020) |
| **Compress Context** | 0.0085 | Total rent ₹32786 in August, Utilities ₹35680 in August |
| **Isolate Context** | 0.0364 | Budget: Paid ₹4185, ₹4575;  |

---

### Query 2: *“What is my investment goal?”*

| Approach | Query Time (s) | Retrieved Results |
| --- | --- | --- |
| **Write Context** | 0.0249 | Invested ₹20359, Invested ₹37643 |
| **Select Context** | 0.4382 | *(empty result in test)* |
| **Compress Context** | 0.0093 | Latest investment ₹20000 SBI, Avg savings ₹500000 |
| **Isolate Context** | 0.0296 | Investment: mutual fund ₹40012, ₹10558 |

---

### Query 3: *“What is my savings balance?”*

| Approach | Query Time (s) | Retrieved Results |
| --- | --- | --- |
| **Write Context** | 0.0201 | Balance ₹436996 (day 2030), ₹467776 (day 2037) |
| **Select Context** | 1.9595 | Balance ₹436996 (day 2030), ₹467776 (day 2037) |
| **Compress Context** | 0.0075 | Avg balance ₹500000, Utilities ₹35680 |
| **Isolate Context** | 0.0235 | Budget: balance ₹641769  |

### Summary

### For financial assistant chatbot

- **Best hybrid:**
    
     Use **Isolate Context** (separate FAISS indexes per domain) for day-to-day queries.
    
    Add **Compress Context** index for fast summaries (“What’s my average rent?”, “Total savings trend?”).
    
    Keep **Write Context** only for fallback when you need exact raw logs.
    
    Drop **Select Context** (too slow at 75k docs).
    

| Approach | Accuracy | Speed | Scalability | Best For |
| --- | --- | --- | --- | --- |
| **Write** | High | Fast | Poor | Exact logs, small–medium dataset |
| **Select** | High | Slow | Very poor | Toy datasets only |
| **Compress** | Medium | Super fast | Excellent | Summaries, trends |
| **Isolate** | High | Fast | Good | Domain-specific queries |
