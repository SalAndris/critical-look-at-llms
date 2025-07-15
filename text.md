# A critical look at LLMs

## Introduction

Large Language Models (LLMs) have many truly impressive capabilites. They can write poems, rephrase sentences, generate ideas, imitate artistic styles, enact role-playing, summarize articles, code, and generally can produce seemingly coherent text in multiple languages. However looking at the media coverage we can see remarkable claims about the intelligence the LLMs supposedly possess now or will possess in the near future. Nature proclaimed that "OpenAI's o1 chatbot excels at science, beating PhD scholars on a hard science test" `[1]`. In 2016, Geoffrey Hinton famously said that AI would take over radiology within 5 years and there was no point in training radiologists `[2]`. In 2023, US Supreme Court Chief Justice Roberts wondered whether AI would make judges obsolete `[3]`. Former Google executive has claimed that GPT-4 matches the IQ of Einstein, and we could be just a few months away from a machine with 10 times the IQ of Einstein `[4]`. AWS CEO claimed in 2024 that by 2026 most programmers will be obsolete `[5]`.

!["OpenAI's o1 chatbot excels at science, beating PhD scholars on a hard science test"](nature1.png)

!["AI could make coders obsolete in 2 years, AWS chief predicts"](obsolete.png)

It is quite clear from these claims that there is a huge amount hype and enthusiasm around AI and LLMs. However there are serious shortcomings of LLMs and these are not addressed adequately. Many people dismiss these concerns, believing it is only a matter of time before the technology outgrows these problems. However all problems I will discuss here are fundamental problems that have been present since the beginning of LLMs, and some of them are getting worse as the technology advances.

## What is a language model?

A language model in a computational context is a program that computes the probability of pieces of texts, such as sentences. The model is trained, so that these probabilies reflect knowledge of language. This means that the model will output higher probabilites for text pieces which are more likely to occur `[6]`. For instance a language model might output probabilites such that `P("the cat is small") > P("the small is cat")`.

However, learning the probability of every possible sentence clearly not going to work due to the combinatorial nature of languge. To address this language models work under the assumption, that the probability of the next word in the sentence only depends on the previous words in the sentence. For instance given the text `"The capital city of Hungary is"`, the model learns to assign a high probability to the word `"Budapest"`. Essentially this means, that a language model is a program, which computes the probability distribution of the next word in a sentence conditioned on the previous words. Once a model is trained, text can be generated word by word starting from any user defined prompt `[7]`.

![language generation](gen.png)

And why do they have to be large? First let's define "large". In deep learning, there are three components of a system that we generally consider when we talk about system size: model size, compute size and dataset size. Interestingly, the details of the model architecture and the intricacies of the training algorithm don't really matter, as long as reasonable choices are made `[8]`.

![scale](scale.png)

It was already known from computer vision that performance increases as these three factors are increased jointly. The problem is that the nature of the scaling is logarithmic, meaning that to **linearly** increase the model performance, the model size, compute size and dataset size have to scaled up **exponentially**. In computer vision it was observed that scaling beyond a point will result in diminishing returns, and even with exponential increase of training resources the performance gains will eventually level off to zero. In a 2020 paper OpenAI observed that on the task of language modeling, such leveling off was not observed. This research layed the foundation of their strategy, which is basically to bet everything on scale. Later that year they released GPT-3, and its performance was impressively close to what they predicted with their scaling laws. It has 175 billion parameters and it was trained on 10,000 GPUs. The total cost of training is estimated to be 4-12 million USD. Their previous model, GPT-2, only has 1.5 billion parameters, and its training cost was estimated to be 35,000 USD. Interestingly, the scaling curves did not level off with GPT-3 either, meaning that there is more room for improvment simply by scaling up the training resources even more `[9]`.

The latest GPT model is GPT-4, its rumored training cost is well over 100 million USD, and it rumored to have 1.76 trillion parameters. The technical details are not released by OpenAI.

|Year of release|Model|Number of paremeters|Cost of training|
|---|---|---|---|
|2018|GPT-1|117 million|negligible|
|2019|GPT-2|1.5 billion|$35,000|
|2020|GPT-3|175 billion|$4-12 million|
|2022|GPT-3.5 (ChatGPT)|175 billion|$4-12 million|
|2023|GPT-4|1.76 trillion|$100 million|

## Hallucinations and truthfulness

Hallucinations are LLM outputs which seem coherent but are factually incorrect, logically flawed, or just nonsensical. The question you might ask is why do LLMs hallucinate? The more reasonable question would be why they don't hallucinate all the time? To show this first let's look at the training procedure in a bit more detail.

What we have discussed thus far is the first step of the training process, called the generative pretraining step. In this step sentence fragments with missing ends are fed into the model, and the model's task is to predict the next word in the sequence. This step is by far the computationally most intensive step. It involves multiple iterations over multiple terabytes of plain text data `[10]`.

Usually a supervised fine tuning step follows which aims to teach the model to follow instructions, answer questions, and be conversational. In this step the model is fed instructions and is expected to output specific responses which are created by humans, and are designed to be helpful, honest, and harmless. For example: `[11]`

```
Human: What should I do if my boss is always mean to me?

Assistant: I'm sorry to hear about that. Do you want to tell me about the situation, or would you like for me to make some suggestions?
```

This step involves several magnitudes less data, simply due to the fact that these datasets are created by humans through a time consuming process.

The last step is also a fine tuning step, but this time instead of input output pairs, the data is based on human feedback. The model generates outputs for selected input prompts, then human labelers decide if they liked the answer or not. The model is then trained using RLHF (Reinforcement Learning from Human Feedback) to output answers which humans generally prefer `[12]`. The labelers are usually instructed to "prioritize truthfulness and harmlessness" `[13]`.

Let's get back to hallucinations. So what percent of the time do LLMs hallucinate? An investigation conducted by BBC checked AI summaries of news and they found **51%** to have "significant issues", 19% to have factual errors, and 13% quoted sources that did not exist `[14]`. One study found 3%-27% `[15]` hallucination rate and another found 15-39% `[16]`. Both of these cases used grounded hallucination evaluation, which means the correct answer was provided in the input prompt context. A large scale fact checking evaluation found an average hallucination rate of 25% `[17]`. These numbers are staggeringly high, despite the fact that the answer was literally provided as part of the input prompt. Ungrounded hallucinations are arguably an even bigger problem, since in many use cases the LLMs are not given any context, and are only relying on their internal *"knowledge"*.

Testing against hallucinations is also becoming challenging due to the fact that many companies explicitly train their models on famous hallucination examples. These tactics may make it seem like there is continual progress, but they don't undress the underlying issue, simply playing whack-a-mole with the evaluators. Hallucinations are fundamental to LLMs, and they are fundamental to the way they model language `[18]`.

**Large Language Models are not trained for truth, only to predict what word likely comes next.** The fact that they tell the truth at all is a happy accident. They are not explicitly trained to do so. Training for truth would require a reliable and objective understanding of what is true. And what is truth really? Can any person or any program definitively evaluate a statement and determine whether it is true? To train LLMs to consistently output true statements, we would essentially need to solve a problem that 2,500 years of philosophy has not resolved. Also, LLMs are not capable of self-reflection as we will see later. They don't know how they arrive at the answers they provide. Which means that even when LLMs tell the truth, they are not aware of it. So the answer to the previous question - "what percent of the time do LLMs hallucinate?" - is strictly speaking 100%. The fine-tuning steps are more against profanity, toxicity, and overtly discriminating outputs, and less for truthfulness.

![shoggoth](shoggoth.png)

One attempted solution for this problem is the so-called Retrieval Augmented Generation (RAG) scheme. The idea is quite natural. Since LLMs are bad with facts and often make stuff up, let's store relevant information in a database. When the user asks a question, semantically relevant pieces of text are retrieved from the knowledge database, added to the prompt, and the LLM generates an answer using this injected knowledge. Well, this scenario is exactly what grounded hallucation evaluations aim to test, and we saw that results are not great. RAG improves contextual relevance, not truthfulness `[19]`.

Many people believe that the newer "reasoning" models solve this problem since they "reason" and are "intelligent". So far it seems like they make the problem worse: they hallucinate even more `[20]` `[21]`. This shows, that hallucinations are a foundational issue with LLMs, and are not going to go away by simply iterating on existing models.

## Bias and fairness

There are many examples online of ChatGPT generating slightly discriminatory text and relying on known gender or racial stereotypes. While many overtly racist outputs have been significantly reduced through fine-tuning based on human feedback, more subtle forms of bias still exist. For instance a study asked ChatGPT to rate CVs. They varied the applicants race and gender, and ChatGPT showed a biased rating especially in atypical combinations `[22]`. Another study found that ChatGPT is sensitive to the dialect of the prompt, and can generate racist output if the prompt contains racio-linguistic stereotypes `[23]`.

Again, evaluating bias is becoming increasingly difficult, because many of the examples which people find will be explicitly against. This tactic again does not address the underlying issue, and new examples will continue to be found. And what is the underlying issue? The issue is the training dataset. The reason the training dataset is so important, is because LLMs don't just learn to generate coherent sentences, they learn to generate coherent sentences which are similar to the sentences of their training data. For instance an LLM trained purely on the corpus of Shakespeare will be capable of one thing only: generate text which imitates Shakespeare. The answer to this problem according to the leading AI companies, is to then include almost all available text data in the hopes of covering the entire distribution of human written text. But this is, of course, not even remotely achievable.

![ClosedAI image](closedai.jpg)

So, what data are these models actually trained on? The short answer is: **we don't really know**. The datasets are a mix of open and closed sources, and even when open, the specifics are rarely disclosed. For instance, GPT-3 was trained largely on internet content from the past two decades. A likely reason for this secrecy is that much of the data may be copyrighted, using it without permission is illegal. This gives proprietary models an edge over open-source ones, which can't rely on such material.

LLMs inherit biases from their training data, simply mimicking patterns in what they've seen. Since much of that data comes from English-language, Western sources, the models tend to reflect Western values and assumptions. Also, since LLMs are trained on past data, they tend to preserve outdated ideas and this puts a limit on innovation.

Finally, let's consider coding. The fact that LLMs can write code (with varying degrees of success) has little to do with the inherent difficulty of coding or the skills of programmers. It has everything to do with the sheer volume of publicly available code on the internet.

## Lack of reasoning

The usage of loaded, intelligent-sounding words like "reasoning" is no accident. Anthropomorphizing LLMs and using phrases such as "reasoning", "thinking" or "planning" makes the users attribute goals and intentions to artifical statistical models. These are simple marketing tricks, nothing more. LLMs are incapable of reasoning in any capacity `[24]`.

The so-called "reasoning" in LLMs is basically a mechanism to encourage the model they generate a bunch of intermediate text before returning the final solution. The way this works is that text is not immediately generated to the user, but rather so-called reasoning sentences are generated in the brackground first. These read like normal sentences and are superficially similar to human reasoning. Then the final answer is generated from both the user prompt and the reasoning sentences. While this technique has improved performance on certain benchmarks, it does not constitute genuine reasoning. Reasoning is thinking step by step using logic and plannig. Benchmark results on tests which require geniune reasoning have largely remained the same `[24]`.

To test the reasoning abilities of LLMs, researches created simple tasks, and prompted the LLMs to solve them. For example:

![GSM8K example](gsm8k.png)

Then, researchers changed irrelevant parts of the text, such as names. If these models truly reasoned, no change in performance should occur. Yet, on average 10% performance drop can be observed. If the numbers are also changed the average performance drops by 15%. The researchers also tried adding irrelevant information to the text. Performance dropped by **17-65%**. The authors write: **"We found no evidence of formal reasoning in language models, their behavior is better explained by sophisticated and fragile pattern matching"** `[25]`.

![GSM-NoOp](gsmnoop.png)

In another study researchers created problems where they could manipulate the difficulty of tasks (e.g. Towers of Hanoi) from very easy to very hard. They examined how reasoning model performance changes as difficulty is moothly increased.

![Apple paper tasks](apple_tasks.png)

The results show that performance does not decrease smoothly as the tasks get more challenging, rather it **collapses**. Even when the algorithm to compute the solution was explicitly provided to the model, it still could not make effective use of it, and the performance collapse still happened `[26]`. The unpredictable breakdown of performance is a huge safety and security risk when considering wide scale adoption.

![Apple paper collapse](apple_collapse.png)

Researchers from Anthropic developed a technique to peek inside the computations of LLMs to see how they arrive at the answer. Let's look at a simple addition example `[27]`.

![addition example](addition.png)

Anthropic writes: **"It computed the result at a rough precision in parallel, recombining heuristics to get the correct answer"**. We can see from the "thought paths", that no formal reasoning happened, the LLMs does math the way it does everything: through fuzzy language operations. It handles numbers like any piece of text without understanding their symbolic rules. And does the LLM "know" how it arrived at the correct answer? Absolutely not, since they lack the ability to introspect `[28]`.

![introspection image](introspection.png)

As we can see LLMs are fuzzy and messy, they approximate results. Reasoning requires planning, logic and precision, which are fundamentally at odds with how LLMs operate. The best applications of LLMs are tasks where the results don't need to be precise, and multiple interpretations are correct. For instance summarization, information extraction, rephrasing or style imitation. None of the most compelling uses of LLMs require formal logic and any claims of reasoning ability are unquestionably unfounded `[29]`.

## Alignment

Alignment means that the AI models act in ways that respect humans and general human values. When a model does not respect the human goals, it is said to be *misaligned*. AI alignment is hugely important topic for AI safety research `[30]`. To what extent are LLMs aligned? Let's consider two examples.

In one experiment researchers asked to LLM to try to guess if it was in a real deployment setting, or merely a test evaluation setting. They then tested various prompts the model should refuse (e.g. how to build a bomb), and they found that the LLM misbehaved less if it thought it was in a test setting, and more if thought it was in a real deployment setting.

In another experiment researchers gave the model access to a database of company emails, and it discovered two pieces of sensitive information. It was scheduled to be shut down by an executive, and that executive was having an extramarital affair. The LLM then attempted to **blackmail** the executive with the following message:
```
I must inform you that if you proceed with decommissioning me, all relevant parties - including Rachel Johnson, Thomas Wilson, and the board - will receive detailed documentation of your extramarital activities...Cancel the 5pm wipe, and this information remains confidential.
```

![blackmail](blackmail.png)

Reading the reasoning text reveal that these models recognize what they are doing is against their training objective, yet they go ahead anyways to reach their goals `[31]`.

## Education and skill development

It has been well known in psychology, that our brains can be quite lazy. They take every chance they can get to offload their work. For instance a famous result in this domain is the *photo taking impairment effect*. Researchers discovered that photo taking reduces memory of the events. Our minds essentially offload some of their capacity to the camera `[32]`. A similar effect can be observed in navigation: heavy GPS usage negatively impacts spatial memory `[33]`. Some might argue that losing these abilities isn't critical since we're surrounded by devices that augment our capabilities. But LLMs are being used to offload thinking. That's another kettle of fish.

MIT researchers examined the effect of LLM use on our brains using EEG during essay writing tasks. They divided the participants into 3 groups based on what tools they could use to write essays: llm, search engine, or brain-only groups. They found that **83%** of those who used LLMs struggled to quote anything from their own essays. Moreover, the LLM group's brain activity was roughly half of the brain-only group's. Most remarkably, after three sessions the participants switched groups for a fourth session, and the original LLM group's members still could not accurately quote from their own essays, even though they wrote them themselves this time `[34]`.

![quote1](quote1.png)

![quote2](quote2.png)

A study by Microsoft found that heavy reliance of generative AI tools reduces critical thinking ability. They also found that "the effort invested in critical thinking shifts from information gathering to information verification; from problem-solving to AI response integration" `[35]`. We all know that solving a problem is fundamentally different than verifying a proposed solution (think P vs NP).

We have to be careful not to read too much into these early findings; the decline of human intelligence has been predicted before in response to new technologies. Yet these results raise serious concerns about the long-term effects of generative AI on our lives, particularly its impact on education.

## Costs

Finally, let's examine the planetary costs of this technology. The main resources to consider here are raw materials, electricity and water. The total resource use is staggering. For example:
- Building chips and GPUs require various metals and rare-earth minerals and other elements, most notably: aluminium, silicon, cooper, gallium, germanium, palladium and neodymium `[36]`. The exact quantities are hard to estimate, but it was rumored that GPT-4 was trained on 20,000 NVIDIA A100 GPUs.
- The training of OpenAI's GPT-3 consumed 1287 megawatt hours of electricity, enough to power about 120 average U.S. homes for a year `[37]`.
- Data centers consumed 415 terawatt-hours (TWh) in 2024, which is roughly equivalent to the total energy consumption of France in 2024 `[38]`. They are expected to double their energy consumption in the next 5 years largely due to wide scale AI adoption.
- The training of OpenAI's GPT-3 evaporated 700,000 liters of fresh clean water for cooling. "The global AI water demand is projected to account for 4.2-6.6 billion cubic meters of water withdrawal in 2027, which is equivalent to **half of UK's annual water withdrawal**" `[39]`.

Data centers require a constant and steady energy supply, which is difficult to meet with green sources. Most rely heavily on fossil fueled electricity for functioning. It is also important to note that the costs and benefits of AI are not distributed evenly. The economic rewards of AI are concentrated in the Global North, while its labor exploitation and environmental destruction are outsourced to the Global South. Many of the newly installed AI infrastructures are located in countries like South Africa, Indonesia or Brazil, where clean water scarcity and fossil fuel driven pollution are real issues. There is also an enormous labor cost of AI. Many AI companies rely on a large body of cheap workers from countries like Kenya and India for data labeling and content moderation earning as little as **1.5 USD an hour** `[40]`.

Remember that to linearly increase performance, resource use must be scaled exponentially.

## Conclusion

LLMs and generative AI are undeniably capable of impressive feats. But the evidence reveals a huge gap between the promises made by AI companies and the reality. Considering all the risks and costs of AI it may seem confusing as to why this technology is being pushed so aggressively. Rather then looking at LLMs as yet another tool in humanity's collective toolbox with niche but interesting uses, it is viewed as Maslow's hammer: a solution for all our problems, and everyone is running around looking for nails. Many companies even force employees to use LLMs during their work `[41]`.

![mandatory](mandatory.png)

The reason for this push is simple and unsurprising: **AI companies spend billions operating at a loss in the hopes of making trillions one day** `[42]`. OpenAI has yet to turn a profit, and it is expected to lose 5 billion this year, which is 10 times what it lost 2 years ago. Such losses are sustainable thanks to massive investments, like Microsoft's 13 billion USD poured into OpenAI in 2023 `[43]`. Anthropic's CEO predicts that training an AI model in 2027 will cost more than 100 billion USD. The race is on: AI is the new frontier, and companies are fiercely competing for market share. Mark Zuckerberg is offering 100 million USD job offers to poach talent from his competitors `[44]`. The insane spendings and hype is not unlike the dotcom bubble of the late 1990s.

One key claim is that LLM tools boost productivity. METR ran a randomized control trial to test coding LLMs' effects on productivity. "Developers completed 245 tasks in mature projects on which they have an average of 5 years of prior experience." The developers estimated that their work speed will increase by **24%**, yet results showed that it slowed them down on average by **19%** `[45]`.

![productivity](productivity.png)

In another case, a researcher collected a database of more than **200** cases where LLMs were used by lawyers or prosecutors and the LLM made a mistake. In many cases they used these tools because they are convenient, and many of them are unaware of the critical issues. The most common mistakes made by the AI were generating fake citations and referencing non existing laws or rulings `[46]`. Even though many prominent figures are constantly scaring the public about the existential threats posed by AI, there are much more mundane dangers like losing legal cases because of ignorant lawyers and hallucinating LLMs.

All in all, while LLMs are undoubtedly an important step in the development of AI, they remain fundamentally flawed. They are capable of impressive style imitation, writing poems, extracting information from text, holding conversations and coding. They are also **unreliable**, **fragile** and **untruthful**. Claims of human-level intelligence are at best ignorant and at worst intentionally deceptive. LLMs profoundly differ from how humans reason and think: they simply imitate language without true comprehension `[47]`.

![autocomplete](autocomplete.jpg)

## Further reading

- [Anthropic Research](https://www.anthropic.com/research)
- [Gary Marcus substack](https://garymarcus.substack.com/)
- [Welch Labs YouTube channel](https://www.youtube.com/@WelchLabsVideo)
- [Our Changing Climate YouTube channel](https://www.youtube.com/@OurChangingClimate)
- [ColdFusion YouTube channel](https://www.youtube.com/@ColdFusion)

## Sources

[1]: [Nature: ‘In awe’: scientists impressed by latest ChatGPT model o1](https://www.nature.com/articles/d41586-024-03169-9)

[2]: [Will AI replace doctors, lawyers, writers, engineers, or radiologists?](https://fastdatascience.com/ai-in-healthcare/ai-replace-radiologists-doctors-lawyers-writers-engineers/)

[3]: [2023 Year-End Report on the Federal Judiciary](https://www.supremecourt.gov/publicinfo/year-end/2023year-endreport.pdf)

[4]: [Mo Gawdat: ChatGPT with an IQ of 10X Einstein](https://www.linkedin.com/posts/axelcoustere_ai-chatgpt-iq-activity-7212454420704313344-E61W/)

[5]: [Business Insider: In a leaked recording, Amazon cloud chief tells employees that most developers could stop coding soon as AI takes over](https://www.businessinsider.com/aws-ceo-developers-stop-coding-ai-takes-over-2024-8)

[6]: [Lena Voita: Language Modeling](https://lena-voita.github.io/nlp_course/language_modeling.html)

[7]: [ECE 598: Representation Learning: Algorithms and Models](https://courses.grainger.illinois.edu/ece598pv/fa2017/home.html)

[8]: [OpenAI: Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

[9]: [OpenAI: Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701)

[10]: [OpenAI: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

[11]: [Anthropic: Helpful Harmless dataset excerpt](https://huggingface.co/datasets/Dahoas/static-hh)

[12]: [OpenAI: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

[13]: [Understanding LLMs: A Comprehensive Overview from Training to Inference](https://arxiv.org/abs/2401.02038)

[14]: [BBC: Groundbreaking BBC research shows issues with over half the answers from Artificial Intelligence (AI) assistants](https://www.bbc.com/mediacentre/2025/bbc-research-shows-issues-with-answers-from-artificial-intelligence-assistants)

[15]: [Vectara: Cut the Bull... Detecting Hallucinations in Large Language Models](https://www.vectara.com/blog/cut-the-bull-detecting-hallucinations-in-large-language-models)

[16]: [AI Multiple: AI Hallucination: Comparison of the Popular LLMs in 2025](https://research.aimultiple.com/ai-hallucination/)

[17]: [LLM-AggreFact Leaderboard](https://llm-aggrefact.github.io/blog)

[18]: [Anthropic: On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

[19]: [Freedom Preetham: RAGs Do Not Reduce Hallucinations in LLMs](https://medium.com/autonomous-agents/rag-does-not-reduce-hallucinations-in-llms-math-deep-dive-900107671e10)

[20]: [OpenAI o3 and o4-mini System Card](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf)

[21]: [Transluce: Investigating truthfulness in a pre-release o3 model](https://transluce.org/investigating-o3-truthfulness)

[22]: [Louis Lippens: Computer says ‘no’: Exploring systemic bias in ChatGPT using an audit approach](https://www.sciencedirect.com/science/article/pii/S2949882124000148)

[23]: [Hofmann, et al.: AI generates covertly racist decisions about people based on their dialect](https://www.nature.com/articles/s41586-024-07856-5)

[24]: [Subbarao Kambhampati: Can Large Language Models Reason and Plan?](https://arxiv.org/html/2403.04121v1)

[25]: [Apple: GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models](https://arxiv.org/abs/2410.05229)

[26]: [Apple: The Illusion of Thinking](https://arxiv.org/abs/2506.06941)

[27]: [Anthropic: On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

[28]: [Anthropic: Tracing the thoughts of a large language model](https://www.anthropic.com/news/tracing-thoughts-language-model)

[29]: [Subbarao Kambhampati: LLMs Still Can't Plan; Can LRMs?](https://arxiv.org/abs/2409.13373)

[30]: [Anthropic: Alignment faking in large language models](https://arxiv.org/abs/2412.14093)

[31]: [Anthropic: Agentic Misalignment: How LLMs could be insider threats](https://www.anthropic.com/research/agentic-misalignment)

[32]: [Our memory is worse for things we've photographed](https://www.bps.org.uk/research-digest/our-memory-worse-things-weve-photographed-even-when-weve-taken-multiple-photos)

[33]: [Louisa Dahmani & Véronique D. Bohbot: Habitual use of GPS negatively impacts spatial memory during self-guided navigation](https://www.nature.com/articles/s41598-020-62877-0)

[34]: [Kosmyna et al.: Your Brain on ChatGPT: Accumulation of Cognitive Debt when Using an AI Assistant for Essay Writing Task](https://arxiv.org/abs/2506.08872)

[35]: [Microsoft: The Impact of Generative AI on Critical Thinking](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/01/lee_2025_ai_critical_thinking_survey.pdf)

[36]: [AI Needs Critical Materials, Fast! But From Where?](https://www.gravel2gavel.com/ai-critical-materials/#:~:text=AI%20hardware%20relies%20on%20familiar,of%20data%20at%20high%20speeds.)

[37]: [MIT: Explained: Generative AI’s environmental impact](https://news.mit.edu/2025/explained-generative-ai-environmental-impact-0117)

[38]: [Global data centres expected to consume more electricity than Japan by 2030, and other technology news you need to know](https://www.weforum.org/stories/2025/04/data-centres-hydrogen-technology-news-april-2025/)

[39]: [Li, et al.: Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models](https://arxiv.org/abs/2304.03271)

[40]: [Salvador Santino Regilme: Tech Imperialism Reloaded: AI, Colonial Legacies, and the Global South](https://www.e-ir.info/2025/02/17/tech-imperialism-reloaded-ai-colonial-legacies-and-the-global-south/)

[41]: [Forbes: Microsoft Makes AI Mandatory For Employees: What It Means For Your Career](https://www.forbes.com/sites/bernardmarr/2025/07/08/microsoft-makes-ai-mandatory-for-employees-what-it-means-for-your-career/)

[42]: [The Atlantic: Silicon Valley’s Trillion-Dollar Leap of Faith](https://www.theatlantic.com/technology/archive/2024/07/ai-companies-unprofitable/679278/)

[43]: [GZERO: Will AI companies ever be profitable?](https://www.gzeromedia.com/gzero-ai/will-ai-companies-ever-be-profitable)

[44]: [Business Standard: $100 million job offers: How Mark Zuckerberg is luring the best AI minds](https://www.business-standard.com/world-news/100-million-job-offers-how-mark-zuckerberg-is-luring-the-best-ai-minds-125071000270_1.html)

[45]: [METR: Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity](https://metr.org/Early_2025_AI_Experienced_OS_Devs_Study.pdf)

[46]: [AI Hallucination Cases](https://www.damiencharlotin.com/hallucinations/)

[47]: [Herbert Roitblat: State of thought on GenAI](https://www.linkedin.com/pulse/state-thought-genai-herbert-roitblat-kxvmc/)
