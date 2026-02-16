# Chain of Thought (CoT) Prompt Examples

A collection of practical prompt templates demonstrating Chain of Thought reasoning.

---

## Zero-Shot CoT (Simplest Approach)

### Basic Template
```
Question: [Your question here]

Let's think step by step.
```

### Example 1: Math Word Problem
```
Question: A bakery makes 150 cookies in the morning and 200 cookies in the afternoon. 
They sell 85% of the morning batch and 70% of the afternoon batch. How many cookies 
are left unsold?

Let's think step by step.
```

**Expected Output:**
```
Step 1: Calculate morning cookies sold
- Morning batch: 150 cookies
- Sold: 150 × 0.85 = 127.5 → 127 cookies (can't sell half a cookie)
- Remaining: 150 - 127 = 23 cookies

Step 2: Calculate afternoon cookies sold
- Afternoon batch: 200 cookies
- Sold: 200 × 0.70 = 140 cookies
- Remaining: 200 - 140 = 60 cookies

Step 3: Total remaining
- Total unsold: 23 + 60 = 83 cookies

Answer: 83 cookies remain unsold.
```

### Example 2: Logic Puzzle
```
Question: If all roses are flowers, and some flowers fade quickly, can we conclude 
that some roses fade quickly?

Let's think step by step.
```

---

## Few-Shot CoT (Providing Examples)

### Template Structure
```
[Example 1 with reasoning]

[Example 2 with reasoning]

Now solve this problem:
[Your actual question]
```

### Example: Arithmetic Reasoning

```
Q: A restaurant had 23 customers. 15 more customers arrived, then 8 customers left. 
How many customers are in the restaurant now?

A: Let me solve this step by step:
Step 1: Start with 23 customers
Step 2: Add 15 customers who arrived: 23 + 15 = 38
Step 3: Subtract 8 customers who left: 38 - 8 = 30
Answer: 30 customers are in the restaurant now.

---

Q: A store had 48 books. They sold 12 books on Monday and 15 books on Tuesday. 
Then they received a shipment of 30 new books. How many books does the store have now?

A: Let me solve this step by step:
Step 1: Start with 48 books
Step 2: Subtract Monday sales: 48 - 12 = 36
Step 3: Subtract Tuesday sales: 36 - 15 = 21
Step 4: Add new shipment: 21 + 30 = 51
Answer: The store now has 51 books.

---

Now solve this problem:
Q: A parking lot had 67 cars. 23 cars left, then 18 cars arrived, and finally 
5 more cars left. How many cars are in the parking lot now?

A:
```

---

## Advanced CoT Patterns

### Self-Consistency CoT
Generate multiple reasoning paths and choose the most consistent answer:

```
Question: If you flip a coin 3 times, what's the probability of getting exactly 2 heads?

Generate 3 different reasoning approaches:

Approach 1: [Combinatorics method]
Approach 2: [List all outcomes method]  
Approach 3: [Probability tree method]

Compare the answers and provide the most reliable solution.
```

### Least-to-Most Prompting
Break complex problems into simpler subproblems:

```
Question: A company's revenue grew 20% in Year 1, then 15% in Year 2, and declined 
10% in Year 3. If they started with $1,000,000, what's their revenue after Year 3?

First, let's break this into simpler questions:
1. What's the revenue after Year 1?
2. What's the revenue after Year 2?
3. What's the revenue after Year 3?

Now solve each step:
```

---

## Domain-Specific CoT Examples

### Code Debugging
```
Problem: This Python function is supposed to find the maximum value in a list, 
but it's not working correctly:

def find_max(numbers):
    max_val = 0
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val


Let's debug this step by step:
1. What does the function currently do?
2. What test cases might break it?
3. What's the root cause of the bug?
4. How should we fix it?
```

### Medical Diagnosis (Educational)
```
Patient presents with: fever (102°F), sore throat, swollen lymph nodes, and fatigue 
for 3 days.

Let's work through the differential diagnosis step by step:
1. What are the key symptoms?
2. What common conditions match these symptoms?
3. What additional questions would help narrow down the diagnosis?
4. What's the most likely diagnosis and why?
```

### Business Strategy
```
Question: Should a small coffee shop invest $50,000 in adding a bakery section?
Current revenue: $200,000/year. Expected bakery revenue: $80,000/year. 
Expected costs: $30,000/year. Payback period goal: 2 years.

Let's analyze this step by step:
1. Calculate net bakery profit
2. Calculate payback period
3. Consider additional factors (foot traffic, competition)
4. Make a recommendation with reasoning
```

---

## Tips for Effective CoT Prompts

### ✅ DO:
- Be explicit: "Let's think step by step" or "Let's break this down"
- Number your steps for clarity
- Show your work for calculations
- State assumptions when needed
- Verify your answer makes sense

### ❌ DON'T:
- Skip intermediate steps
- Mix up the order of operations
- Forget units (e.g., dollars, meters)
- Make logical jumps without explanation

---

## Prompt Variations

Different ways to trigger CoT reasoning:

```
"Let's think step by step."
"Let's break this down."
"Let's solve this systematically."
"Let's approach this methodically."
"Let's work through this carefully."
"First, let's understand... Then..."
"Let's reason through this."
```

---

## Testing Your CoT Prompts

Try your prompt on these benchmark problems:

**Easy:**
```
If John has 3 apples and gives 1 to Mary, how many does he have left?
```

**Medium:**
```
A train travels 60 miles in 1.5 hours. At this rate, how long will it take to 
travel 150 miles?
```

**Hard:**
```
In a class of 30 students, 18 play soccer, 15 play basketball, and 8 play both. 
How many students play neither sport?
```

## When to Use CoT

**✓ Best for:**
- Multi-step math problems
- Logic puzzles
- Complex reasoning tasks
- Problems requiring calculations
- Situations where you need to show your work

**✗ Not ideal for:**
- Simple factual questions ("What's the capital of France?")
- Creative writing (can make it too mechanical)
- Tasks requiring external information (use ReAct instead)


## Template: Build Your Own CoT Prompt

```
Question: [Your specific problem]

[Optional: Provide 1-2 examples with reasoning if few-shot]

Let's approach this step by step:

Step 1: [Identify what we know]

Step 2: [Determine what we need to find]

Step 3: [Apply relevant method/formula]

Step 4: [Calculate/reason through]

Step 5: [Verify the answer makes sense]

Therefore, [final answer]
```

---

**Pro Tip:** Combine with other techniques like "show your work," "explain your reasoning," or "if you're unsure, say so" for even better results.