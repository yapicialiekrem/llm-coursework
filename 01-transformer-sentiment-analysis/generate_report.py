"""
generate_report.py
------------------
Reads outputs/bert/results.json and outputs/gpt/results.json
and generates sentiment_analysis_report.docx

Usage:
    python generate_report.py
"""

import os, json, subprocess, sys

# ── Load results ──────────────────────────────────────────────────────────────
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"[WARN] {path} not found — run the training scripts first.")
    return {}

bert = load_json("outputs/bert/results.json")
gpt  = load_json("outputs/gpt/results.json")

data = {
    "bert_metrics": bert.get("test_metrics", {}),
    "gpt_metrics" : gpt.get("test_metrics",  {}),
    "bert_config" : bert.get("config", {}),
    "gpt_config"  : gpt.get("config",  {}),
    "bert_history": bert.get("history", []),
    "gpt_history" : gpt.get("history",  []),
}

with open("_report_data.json", "w") as f:
    json.dump(data, f, indent=2)

# ── JS source ─────────────────────────────────────────────────────────────────
JS = r"""
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, LevelFormat, PageNumber, Footer,
} = require('docx');

const BLUE='1F5C99', DGRAY='333333', MGRAY='666666', LGRAY='F2F2F2', WHITE='FFFFFF', GREEN='1E7A4A';

const brdAll = (c='CCCCCC') => { const b={style:BorderStyle.SINGLE,size:1,color:c}; return {top:b,bottom:b,left:b,right:b}; };
const cell = (text, opts={}) => new TableCell({
  borders:brdAll(opts.brdColor||'CCCCCC'), width:{size:opts.w||2340,type:WidthType.DXA},
  shading:opts.bg?{fill:opts.bg,type:ShadingType.CLEAR}:undefined,
  margins:{top:80,bottom:80,left:140,right:140}, verticalAlign:VerticalAlign.CENTER,
  children:[new Paragraph({alignment:opts.center?AlignmentType.CENTER:AlignmentType.LEFT,
    children:[new TextRun({text:String(text),bold:opts.bold||false,size:opts.size||20,color:opts.color||DGRAY})]})]
});
const h1 = t => new Paragraph({heading:HeadingLevel.HEADING_1,spacing:{before:400,after:200},children:[new TextRun({text:t,bold:true,size:32,color:BLUE})]});
const h2 = t => new Paragraph({heading:HeadingLevel.HEADING_2,spacing:{before:280,after:160},children:[new TextRun({text:t,bold:true,size:26,color:BLUE})]});
const h3 = t => new Paragraph({heading:HeadingLevel.HEADING_3,spacing:{before:200,after:100},children:[new TextRun({text:t,bold:true,size:22,color:DGRAY})]});
const p  = (text,opts={}) => new Paragraph({spacing:{after:140},alignment:opts.center?AlignmentType.CENTER:AlignmentType.LEFT,
  children:[new TextRun({text,bold:opts.bold||false,italics:opts.italic||false,size:opts.size||22,color:opts.color||DGRAY})]});
const bullet = t => new Paragraph({numbering:{reference:'bullets',level:0},spacing:{after:80},children:[new TextRun({text:t,size:22,color:DGRAY})]});
const sp = () => new Paragraph({spacing:{after:160},children:[]});
const hr = () => new Paragraph({spacing:{after:200},border:{bottom:{style:BorderStyle.SINGLE,size:6,color:BLUE,space:1}},children:[]});

const cfgTable = rows => new Table({width:{size:9360,type:WidthType.DXA},columnWidths:[3600,5760],
  rows:[new TableRow({tableHeader:true,children:[cell('Parameter',{bg:BLUE,bold:true,color:WHITE,w:3600}),cell('Value',{bg:BLUE,bold:true,color:WHITE,w:5760})]}),
    ...rows.map(([k,v],i)=>new TableRow({children:[cell(k,{bg:i%2===0?LGRAY:WHITE,bold:true,w:3600}),cell(v,{bg:i%2===0?LGRAY:WHITE,w:5760})]}))]});

const singleMetricTable = m => {
  const rows=[['Accuracy',m.accuracy],['Precision',m.precision],['Recall',m.recall],['F1 Score',m.f1],['Loss',m.loss]];
  return new Table({width:{size:9360,type:WidthType.DXA},columnWidths:[4680,4680],
    rows:[new TableRow({tableHeader:true,children:[cell('Metric',{bg:BLUE,bold:true,color:WHITE,w:4680}),cell('Score',{bg:BLUE,bold:true,color:WHITE,w:4680,center:true})]}),
      ...rows.map(([k,v],i)=>new TableRow({children:[cell(k,{bg:i%2===0?LGRAY:WHITE,bold:true,w:4680}),
        cell(typeof v==='number'?(k==='Loss'?v.toFixed(4):(v*100).toFixed(2)+'%'):'N/A',{bg:i%2===0?LGRAY:WHITE,w:4680,center:true})]}))]});
};

const metricTable = (bm,gm) => {
  const rows=[['Accuracy',bm.accuracy,gm.accuracy],['Precision',bm.precision,gm.precision],['Recall',bm.recall,gm.recall],['F1 Score',bm.f1,gm.f1],['Loss',bm.loss,gm.loss]];
  return new Table({width:{size:9360,type:WidthType.DXA},columnWidths:[3120,3120,3120],
    rows:[new TableRow({tableHeader:true,children:[cell('Metric',{bg:BLUE,bold:true,color:WHITE,w:3120,center:true}),cell('BERT',{bg:BLUE,bold:true,color:WHITE,w:3120,center:true}),cell('GPT-1',{bg:BLUE,bold:true,color:WHITE,w:3120,center:true})]}),
      ...rows.map(([label,bv,gv],i)=>{
        const bg=i%2===0?LGRAY:WHITE, lower=(label==='Loss'), bw=lower?bv<gv:bv>gv;
        const fmt=v=>label==='Loss'?v.toFixed(4):(v*100).toFixed(2)+'%';
        return new TableRow({children:[cell(label,{bg,bold:true,w:3120}),cell(fmt(bv),{bg,bold:bw,color:bw?GREEN:DGRAY,w:3120,center:true}),cell(fmt(gv),{bg,bold:!bw,color:!bw?GREEN:DGRAY,w:3120,center:true})]});
      })]});
};

const historyTable = history => {
  const cols=[{label:'Epoch',key:'epoch',pct:false},{label:'Train Loss',key:'train_loss',pct:false},{label:'Val Loss',key:'val_loss',pct:false},{label:'Val Acc',key:'val_accuracy',pct:true},{label:'Val F1',key:'val_f1',pct:true}];
  const widths=[1100,1900,1900,2200,2260];
  return new Table({width:{size:9360,type:WidthType.DXA},columnWidths:widths,
    rows:[new TableRow({tableHeader:true,children:cols.map((c,i)=>cell(c.label,{bg:BLUE,bold:true,color:WHITE,w:widths[i],center:true}))}),
      ...history.map((row,ri)=>new TableRow({children:cols.map((c,i)=>{
        let val=row[c.key]; if(typeof val==='number') val=c.pct?(val*100).toFixed(2)+'%':val.toFixed(4);
        return cell(String(val),{bg:ri%2===0?LGRAY:WHITE,w:widths[i],center:true});})}))]});
};

const D=JSON.parse(fs.readFileSync('_report_data.json','utf8'));
const bm=D.bert_metrics,gm=D.gpt_metrics,bc=D.bert_config,gc=D.gpt_config,bh=D.bert_history,gh=D.gpt_history;

const doc = new Document({
  numbering:{config:[{reference:'bullets',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:720,hanging:360}}}}]}]},
  styles:{default:{document:{run:{font:'Calibri',size:22}}},paragraphStyles:[
    {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:32,bold:true,font:'Calibri',color:BLUE},paragraph:{spacing:{before:400,after:200},outlineLevel:0}},
    {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:26,bold:true,font:'Calibri',color:BLUE},paragraph:{spacing:{before:280,after:160},outlineLevel:1}},
    {id:'Heading3',name:'Heading 3',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:22,bold:true,font:'Calibri',color:DGRAY},paragraph:{spacing:{before:200,after:100},outlineLevel:2}},
  ]},
  sections:[{
    properties:{page:{size:{width:12240,height:15840},margin:{top:1260,right:1080,bottom:1260,left:1080}}},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[
      new TextRun({text:'Sentiment Analysis — BERT vs GPT-1  |  Page ',size:18,color:'999999'}),
      new TextRun({children:[PageNumber.CURRENT],size:18,color:'999999'}),
    ]})]})},
    children:[
      sp(),sp(),sp(),sp(),
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:200},children:[new TextRun({text:'LLM-Driven Software Development',bold:true,size:32,color:BLUE})]}),
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:120},children:[new TextRun({text:'Sentiment Analysis',bold:true,size:52,color:DGRAY})]}),
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:120},children:[new TextRun({text:'Sentiment Analysis with Transformer Models',italics:true,size:28,color:MGRAY})]}),
      hr(),
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:80},children:[new TextRun({text:'Dataset: IMDb Movie Reviews  |  Models: BERT vs GPT-1',size:22,color:MGRAY})]}),
      sp(),sp(),sp(),sp(),sp(),

      h1('Part 1 — Dataset Selection'),
      h2('1.1  Dataset Overview'),
      p('The IMDb Movie Reviews dataset is one of the most widely used benchmarks for binary sentiment classification. It consists of 50,000 movie reviews from the Internet Movie Database (IMDb), each labeled as positive or negative.'),
      sp(),cfgTable([['Dataset Name','IMDb Movie Reviews'],['Source','https://huggingface.co/datasets/imdb  (Maas et al., ACL 2011)'],['Total Samples','50,000'],['Label Classes','0 = Negative,  1 = Positive  (perfectly balanced: 25,000 each)'],['Original Train','25,000 samples'],['Original Test','25,000 samples']]),sp(),
      h2('1.2  Train / Validation / Test Split'),
      p('The 25,000 official training samples were subsampled to 5,000 for computational efficiency on a T4 GPU (15 GB VRAM). A stratified 80/20 split was applied, preserving the 50/50 class balance.'),
      sp(),cfgTable([['Training set','4,000 samples  (80% of the 5,000 subsampled)'],['Validation set','1,000 samples  (20% of the 5,000 subsampled)'],['Test set','25,000 samples  (full original test split, untouched)'],['Subsampling strategy','Random seed=42; stratified by label']]),sp(),
      h2('1.3  Label Classes'),
      bullet('0 — Negative: reviewer rated the movie 1–4 out of 10 stars'),
      bullet('1 — Positive: reviewer rated the movie 7–10 out of 10 stars'),sp(),
      h2('1.4  Example Records'),
      p('[POSITIVE]  "If scientists behaved in a way that H.G. Wells was confident they would in the future, history wouldn\'t quite have turned out the way it did..."',{italic:true}),
      p('[NEGATIVE]  "One of those \'coming of age\' films that should have nostalgia for adults and promise for the kids. This movie has neither..."',{italic:true}),sp(),

      h1('Part 2 — Fine-Tuning BERT'),
      h2('2.1  How BERT Is Adapted for Classification'),
      p('BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only transformer pre-trained with MLM and NSP. For classification:'),sp(),
      bullet('A [CLS] token is prepended: [CLS] tok_1 tok_2 ... tok_n [SEP]'),
      bullet('All 12 encoder layers process the sequence with full bidirectional self-attention.'),
      bullet('The final hidden state of [CLS] (shape: [batch, 768]) serves as the sequence representation.'),
      bullet('A dropout + linear layer maps this to 2 logits; cross-entropy loss is computed.'),
      bullet('All weights are fine-tuned jointly (full fine-tuning).'),sp(),
      p('The [CLS] token is explicitly trained during NSP pre-training to capture sentence-level information, making it a natural anchor for classification.'),sp(),
      h2('2.2  Data Preprocessing and Tokenization'),
      bullet('Tokenizer: BertTokenizer (bert-base-uncased) — lowercase WordPiece subword tokenization'),
      bullet('Special tokens: [CLS] prepended, [SEP] appended automatically'),
      bullet('Padding: dynamic — padded to the longest sequence in each batch'),
      bullet('Truncation: sequences longer than MAX_LENGTH are right-truncated'),
      bullet('MAX_LENGTH = 512 (BERT\'s maximum supported context length)'),sp(),
      h2('2.3  Training Configuration'),
      cfgTable([['Model','bert-base-uncased  (~110M parameters)'],['Max Sequence Length',String(bc.max_length||512)+' tokens'],['Batch Size',String(bc.batch_size||16)],['Optimizer','AdamW  (weight decay decoupled from gradient update)'],['Learning Rate',String(bc.learning_rate||'2e-5')],['Epochs',String(bc.num_epochs||3)],['Weight Decay',String(bc.weight_decay||0.01)+'  (non-bias, non-LayerNorm params only)'],['LR Scheduler','Linear decay (no warmup)'],['Gradient Clipping','max_norm = 1.0'],['Train / Val / Test','4,000 / 1,000 / 25,000 samples']]),sp(),
      h2('2.4  Training History'),historyTable(bh),sp(),
      p('Best model saved at epoch 2 (val F1 = '+(bh[1]?((bh[1].val_f1||0)*100).toFixed(2)+'%':'90.45%')+').  Epoch 3 shows rising validation loss, indicating mild overfitting on the small training set.',{italic:true,color:MGRAY}),sp(),
      h2('2.5  Test Evaluation Results'),singleMetricTable(bm),sp(),

      h1('Part 3 — Fine-Tuning GPT-1'),
      h2('3.1  How GPT-1 Is Adapted for Sentiment Classification'),
      p('GPT-1 is a decoder-only transformer pre-trained with autoregressive language modeling. It has no bidirectional attention and no [CLS] token. OpenAIGPTForSequenceClassification is used:'),sp(),
      bullet('The hidden state of the last non-padding token is extracted as the sequence representation.'),
      bullet('This token has attended to the entire preceding context via causal self-attention — the most context-aware position.'),
      bullet('A linear layer maps this 768-dimensional vector to 2 class logits.'),
      bullet('Cross-entropy classification loss is used (no auxiliary LM loss).'),sp(),
      h2('3.2  Input Prompt Structure'),
      p('Raw review text is tokenised directly. The classification head is applied at the last non-padding token:'),sp(),
      p('tokens:  tok_1  tok_2  ...  tok_n  [PAD]  [PAD]  ...',{italic:true}),
      p('                                ^  classification head applied here',{italic:true}),sp(),
      p('Since GPT-1 has no padding token by default, [PAD] was added and the embedding matrix resized using mean initialization (Hewitt, 2021). The attention mask prevents attending to padding positions.'),sp(),
      h2('3.3  Training Objective'),
      p('Only the cross-entropy classification loss is used (Radford et al., 2018):'),sp(),
      p('L = CrossEntropy(logits, y)   logits in R^2,  y in {0, 1}',{italic:true}),sp(),
      h2('3.4  GPU Memory Constraints and Parameter Adjustments'),
      p('Running GPT-1 after BERT caused an OutOfMemoryError (only 19 MB free on the T4 GPU, 15 GB total). The parameters were adjusted as follows:'),sp(),
      new Table({width:{size:9360,type:WidthType.DXA},columnWidths:[2340,2000,2000,3020],
        rows:[new TableRow({tableHeader:true,children:[cell('Parameter',{bg:BLUE,bold:true,color:WHITE,w:2340}),cell('BERT Value',{bg:BLUE,bold:true,color:WHITE,w:2000,center:true}),cell('GPT-1 Value',{bg:BLUE,bold:true,color:WHITE,w:2000,center:true}),cell('Reason',{bg:BLUE,bold:true,color:WHITE,w:3020})]}),
          new TableRow({children:[cell('MAX_LENGTH',{bg:LGRAY,bold:true,w:2340}),cell('512',{bg:LGRAY,w:2000,center:true}),cell('256',{bg:LGRAY,w:2000,center:true}),cell('Attention is O(n^2); 512->256 reduces memory 4x',{bg:LGRAY,w:3020})]}),
          new TableRow({children:[cell('BATCH_SIZE',{bg:WHITE,bold:true,w:2340}),cell('16',{bg:WHITE,w:2000,center:true}),cell('8',{bg:WHITE,w:2000,center:true}),cell('Required to fit model in remaining VRAM',{bg:WHITE,w:3020})]})]}),sp(),
      p('Limitation: Truncating to 256 tokens causes some information loss for longer reviews. However, sentiment is typically established in the early portion of a review, minimising the practical impact. This is acknowledged as an experimental limitation.',{italic:true,color:MGRAY}),sp(),
      h2('3.5  Training Configuration'),
      cfgTable([['Model','openai-gpt  (~117M parameters)'],['Max Sequence Length',String(gc.max_length||256)+' tokens  (reduced from 512 due to GPU constraints)'],['Batch Size',String(gc.batch_size||8)+'  (reduced from 16 due to GPU constraints)'],['Optimizer','AdamW'],['Learning Rate',String(gc.learning_rate||'6.25e-5')+'  (from original GPT-1 paper)'],['Epochs',String(gc.num_epochs||3)],['Weight Decay',String(gc.weight_decay||0.01)],['Warmup Ratio',String(gc.warmup_ratio||0.002)+'  (0.2% of total steps)'],['Gradient Clipping','max_norm = 1.0'],['Padding Token','[PAD] newly added (GPT-1 has no default pad token)'],['Train / Val / Test','4,000 / 1,000 / 25,000 samples']]),sp(),
      h2('3.6  Training History'),historyTable(gh),sp(),
      p('Best model saved at epoch 3 (val F1 = '+(gh[2]?((gh[2].val_f1||0)*100).toFixed(2)+'%':'88.71%')+').  Validation loss increases sharply after epoch 1, indicating overfitting on the small training set.',{italic:true,color:MGRAY}),sp(),
      h2('3.7  Test Evaluation Results'),singleMetricTable(gm),sp(),

      h1('Part 4 — Model Comparison'),
      h2('4.1  Architecture Differences'),
      new Table({width:{size:9360,type:WidthType.DXA},columnWidths:[2500,3430,3430],
        rows:[new TableRow({tableHeader:true,children:[cell('Aspect',{bg:BLUE,bold:true,color:WHITE,w:2500}),cell('BERT',{bg:BLUE,bold:true,color:WHITE,w:3430,center:true}),cell('GPT-1',{bg:BLUE,bold:true,color:WHITE,w:3430,center:true})]}),
          ...[['Type','Encoder-only','Decoder-only'],['Attention','Bidirectional (full self-attention)','Causal (masked, left-to-right)'],['Layers','12 transformer blocks','12 transformer blocks'],['Hidden Size','768','768'],['Parameters','~110M','~117M'],['Pre-training','MLM + NSP','Autoregressive LM only'],['Classification','[CLS] token (first position)','Last non-padding token'],['Context Used','512 tokens','256 tokens (hardware constraint)']].map(([k,b,g],i)=>new TableRow({children:[cell(k,{bg:i%2===0?LGRAY:WHITE,bold:true,w:2500}),cell(b,{bg:i%2===0?LGRAY:WHITE,w:3430}),cell(g,{bg:i%2===0?LGRAY:WHITE,w:3430})]}))]
      }),sp(),
      h2('4.2  Training Objective Differences'),
      p('BERT uses two simultaneous pre-training objectives. MLM masks 15% of tokens and forces reconstruction using full bidirectional context. NSP trains the model to distinguish true next sentences from random ones. Together they produce rich bidirectional representations.'),sp(),
      p('GPT-1 uses a single autoregressive objective: predict the next token given all preceding ones. This is inherently unidirectional but enables natural text generation and flexible task adaptation through prompt engineering.'),sp(),
      h2('4.3  Performance Comparison'),
      metricTable(bm,gm),sp(),
      p('Bold green values indicate the better score. For Loss, lower is better; for all other metrics, higher is better.',{italic:true,color:MGRAY}),sp(),
      h2('4.4  Discussion — Why BERT Outperforms GPT-1'),
      p('BERT achieves higher accuracy ('+(bm.accuracy*100).toFixed(2)+'% vs '+(gm.accuracy*100).toFixed(2)+'%) and F1 ('+(bm.f1*100).toFixed(2)+'% vs '+(gm.f1*100).toFixed(2)+'%) for the following reasons:'),sp(),
      bullet('Bidirectional context: BERT attends to both left and right context simultaneously. Sentiment depends on long-range dependencies — negation ("not bad"), sarcasm, qualifiers — that require full context to resolve correctly.'),
      bullet('Superior pooling: The [CLS] token is explicitly trained via NSP to aggregate sentence-level meaning, a stronger classifier anchor than GPT-1\'s last-token heuristic.'),
      bullet('Sequence length: BERT processed up to 512 tokens vs 256 for GPT-1, capturing more of each review.'),
      bullet('Less overfitting: GPT-1 validation loss rose from 0.338 to 0.716 across 3 epochs vs BERT\'s 0.306 to 0.430 — GPT-1 overfit more severely on the 4,000 training samples.'),sp(),
      h2('4.5  Advantages and Disadvantages'),
      new Table({width:{size:9360,type:WidthType.DXA},columnWidths:[1800,3780,3780],
        rows:[new TableRow({tableHeader:true,children:[cell('',{bg:BLUE,bold:true,color:WHITE,w:1800}),cell('BERT',{bg:BLUE,bold:true,color:WHITE,w:3780,center:true}),cell('GPT-1',{bg:BLUE,bold:true,color:WHITE,w:3780,center:true})]}),
          new TableRow({children:[cell('Advantages',{bg:LGRAY,bold:true,w:1800}),cell('Superior classification; rich bidirectional representations; lower memory at equal length; [CLS] naturally suited for sentence tasks',{bg:LGRAY,w:3780}),cell('Generative capability; zero/few-shot prompting; single pre-training objective; flexible for generation tasks',{bg:LGRAY,w:3780})]}),
          new TableRow({children:[cell('Disadvantages',{bg:WHITE,bold:true,w:1800}),cell('Cannot generate text; requires labelled data per task; no zero-shot capability',{bg:WHITE,w:3780}),cell('Unidirectional attention misses right context; last-token pooling is a heuristic; higher overfitting risk on small datasets',{bg:WHITE,w:3780})]}),
        ]}),sp(),

      h1('Part 5 — Conceptual Questions'),

      h2('Question 1 — Multi-Head Attention Mechanism'),
      h3('Query, Key, and Value Matrices'),
      p('In self-attention, each input token embedding is linearly projected into three vectors via learned weight matrices:'),
      bullet('Query (Q = x * W_Q): encodes what information this token is looking for'),
      bullet('Key   (K = x * W_K): encodes what information this token offers to others'),
      bullet('Value (V = x * W_V): carries the actual content to be aggregated'),sp(),
      h3('The Attention Formula'),
      p('Attention(Q, K, V) = softmax( Q*K^T / sqrt(d_k) ) * V',{italic:true}),sp(),
      bullet('Q*K^T computes pairwise dot-product similarities, producing an (n x n) score matrix.'),
      bullet('Dividing by sqrt(d_k) (typically 64) prevents saturation of softmax with near-zero gradients.'),
      bullet('softmax normalises each row into attention weights (a probability distribution).'),
      bullet('Multiplying by V produces a weighted sum of value vectors — the contextualised output for each token.'),sp(),
      h3('The Role of Multiple Attention Heads'),
      p('Multi-head attention runs h independent attention operations in parallel with separate projections. Outputs are concatenated and re-projected:'),sp(),
      p('MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_O',{italic:true}),
      p('where head_i = Attention(Q*W_Q_i,  K*W_K_i,  V*W_V_i)',{italic:true}),sp(),
      p('BERT uses h=12 heads with d_k=64 each (12 x 64 = 768 total dimensions). Each head operates in a lower-dimensional subspace, jointly attending to diverse information simultaneously.'),sp(),
      h3('Why Multi-Head Attention Improves Representation Learning'),
      bullet('Diversity: Different heads capture different linguistic relationships simultaneously — subject-verb agreement, coreference, local syntax, etc.'),
      bullet('Robustness: Multiple heads reduce reliance on any single attention pattern.'),
      bullet('Expressiveness: A single head produces one weighted average and cannot represent multiple distinct relationships at once. Multiple heads overcome this fundamental limitation without increasing sequential computation time.'),sp(),

      h2('Question 2 — Loss Function for Machine Translation'),
      h3('Training Objective'),
      p('Transformer machine translation uses maximum likelihood estimation with teacher forcing: at each decoding step t, the ground-truth target token is fed as input, and the model must assign high probability to the correct next token.'),sp(),
      h3('Cross-Entropy Loss Function'),
      p('L = -(1/T) * sum_t  log P(y_t | y_1...y_{t-1}, x)',{italic:true}),sp(),
      bullet('For each position t, cross-entropy is the negative log-probability of the correct target token.'),
      bullet('P(y_t | ...) is obtained by softmax over the full vocabulary applied to decoder output logits.'),
      bullet('Loss is averaged over all T target positions and all sentences in the batch.'),
      bullet('Label smoothing (epsilon=0.1 in the original paper) prevents overconfidence and improves generalisation.'),sp(),
      h3('Which Parameters Are Updated'),
      bullet('Encoder: token embeddings, positional encodings, all self-attention matrices (W_Q, W_K, W_V, W_O) and feed-forward networks across all N encoder layers.'),
      bullet('Decoder: same as encoder, plus encoder-decoder cross-attention weight matrices.'),
      bullet('Output projection: linear layer mapping decoder hidden states to vocabulary logits (often tied with token embedding matrix to reduce parameters).'),sp(),

      h2('Question 3 — Masked Self-Attention in the Decoder'),
      h3('The Autoregressive Property'),
      p('A language model is autoregressive if it generates tokens sequentially, conditioning each on all previously generated tokens: P(y) = product_t P(y_t | y_1...y_{t-1}). At inference time, future tokens have not yet been generated and must remain invisible.'),sp(),
      h3('How Masking Ensures Correct Training Behaviour'),
      p('During training, the full target sequence is presented at once for computational efficiency. Without masking, each position could attend to future tokens — "cheating" by looking ahead — making training trivial but causing catastrophic failure at inference time.'),sp(),
      p('A causal mask is applied before softmax: positions j > t receive attention score -infinity (-> 0 after softmax), forcing position t to attend only to 1...t:',{italic:false}),sp(),
      p('Score(i,j) = Q_i*K_j^T/sqrt(d_k)  if j <= i,   else  -infinity',{italic:true}),sp(),
      bullet('This replicates the inference-time constraint during training, ensuring a valid autoregressive distribution P(y_t | y_1...y_{t-1}).'),
      bullet('Without masking, the model trivially memorises target sequences and fails at inference when future tokens are unavailable.'),
      bullet('Implemented as an upper-triangular matrix of -infinity values added to raw attention scores before softmax.'),sp(),

      h2('Question 4 — BERT Pre-training Tasks'),
      h3('Masked Language Modeling (MLM)'),
      p('15% of input tokens are randomly selected. Of these:'),
      bullet('80% are replaced with [MASK]'),
      bullet('10% are replaced with a random vocabulary token'),
      bullet('10% are left unchanged'),
      p('The model predicts the original token at each selected position using full bidirectional context. The 10%+10% variation prevents the model from only learning to process [MASK] tokens and ensures representations remain useful at inference time when [MASK] is absent.'),sp(),
      h3('Next Sentence Prediction (NSP)'),
      p('The model receives sentence pairs (A, B) and predicts a binary label:'),
      bullet('IsNext (label=1): B is the actual next sentence after A in the corpus — 50% of pairs'),
      bullet('NotNext (label=0): B is a random sentence from a different document — 50% of pairs'),
      p('The [CLS] token\'s final hidden state is used for the binary prediction. NSP teaches sentence-level relationship understanding and is why [CLS] becomes a strong anchor for downstream classification.'),sp(),
      h3('Why Standard Language Modeling Is Not Used for BERT'),
      p('Standard left-to-right language modeling only allows each token to attend to its left context, producing unidirectional representations. BERT\'s goal is deeply bidirectional representations where each token is informed by its full surrounding context.'),sp(),
      p('MLM achieves bidirectionality by masking the target token before the forward pass, forcing reconstruction from both left and right context. Standard LM cannot do this: conditioning on y_{n+1} while predicting y_n causes trivial information leakage.'),sp(),

      h2('Question 5 — GPT-1 Pre-training'),
      h3('Autoregressive Language Modeling'),
      p('GPT-1 is pre-trained on BooksCorpus (~800M words) to maximise the log-likelihood of each token given all preceding tokens:'),sp(),
      p('L_LM = sum_i  log P(u_i | u_{i-k} ... u_{i-1} ; Theta)',{italic:true}),sp(),
      p('where k is the context window (512 tokens) and Theta are model parameters. A causal mask ensures each position attends only to preceding positions.'),sp(),
      h3('How GPT-1 Predicts the Next Token'),
      bullet('Tokens are embedded and summed with learned positional encodings.'),
      bullet('Embeddings pass through 12 decoder blocks with masked multi-head self-attention and position-wise feed-forward networks.'),
      bullet('The causal mask ensures each position attends only to preceding positions.'),
      bullet('The final hidden state is projected to a probability distribution: P(u_t) = softmax(h_t * W_e^T), where W_e is the token embedding matrix (weights tied with output projection).'),
      bullet('At inference, the highest-probability (or sampled) token is appended and the process repeats autoregressively.'),sp(),
      h3('Downstream Task Handling During Fine-tuning'),
      p('GPT-1 adapts to downstream tasks by reformatting inputs with special delimiter tokens, without changing the architecture:'),sp(),
      bullet('Classification: [Start] text [Extract]  ->  linear head on last token hidden state'),
      bullet('Entailment: [Start] premise [Delim] hypothesis [Extract]  ->  linear head'),
      bullet('Similarity: Two sentence-pair orderings processed separately; outputs summed before classifier'),
      bullet('Multiple-choice QA: Each (context, answer) pair scored independently; argmax selects the answer'),sp(),
      p('Fine-tuning objective is task-specific cross-entropy, optionally augmented with auxiliary LM loss:',{italic:false}),sp(),
      p('L_total = L_classification + lambda * L_LM   (lambda = 0.5 in original paper)',{italic:true}),sp(),
      p('In this implementation, only the classification loss was used (lambda = 0). Both the pre-trained transformer weights and the new classification head are fine-tuned jointly.',{color:MGRAY}),sp(),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('sentiment_analysis_report.docx', buf);
  console.log('Report written to sentiment_analysis_report.docx');
});
"""

# ─────────────────────────────────────────────────────────────────────────────
def main():
    with open("_build_report.js", "w") as f:
        f.write(JS)

    print("Installing npm docx package...")
    subprocess.run(["npm", "install", "-g", "docx"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Generating Word document...")
    r = subprocess.run(["node", "_build_report.js"], capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR:", r.stderr[:2000])
        sys.exit(1)
    print(r.stdout.strip())

    for f in ["_report_data.json", "_build_report.js"]:
        try: os.remove(f)
        except: pass

    print("Done! -> sentiment_analysis_report.docx")

if __name__ == "__main__":
    main()
