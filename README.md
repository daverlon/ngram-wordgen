<h1>Word generator using N-gram probabilities</h1>

<p>Word generator using <a href="https://en.wikipedia.org/wiki/N-gram">N-gram</a> probabilities written in Python using only numpy.</p>

<p>Based on <a href="https://github.com/karpathy/makemore" target="_blank">Karpathy's</a> bigram model shown in his makemore lectures.</p>

<p>Example usage:</p>
<pre><p>python3 ngram.py -fdatasets/names.txt -n6 -N30 |> out.txt</p></pre>
<ul>
    <li>-fdatasets/names.txt (use the file located at ./datasets/names.txt)</li>
    <li>-n6 (6-gram model) - required</li>
    <li>-N30 (generate 30 names) - required</li>
    <li>|> out.txt (save generated words into a file, e.g. out.txt) - optional</li>
</ul>
<p>Words with a "✓" already exist in the dataset used.</p>

<br>

<img src="preview.png" width="400px">

<br>

<p>Dependencies:</p>
<ul>
    <li>numpy (arrays, multinomial)</li>
    <li>tqdm (loading bars)</li>
</ul>

<br>

<p>Todo:</p>
<ul>
    <li>Option to save probabilities</li>
    <li>Option to only count non-existing unique generations</li>
    <s><li>Automatic <b>n-gram</b> probability distributions generator</li></s>
</ul>

<br>

<p>Thanks:</p>
<ul>
<li><a href="https://www.ssa.gov/oact/babynames/" target="_blank">SSA.gov baby names dataset</a>
<li><a href="https://github.com/first20hours/google-10000-english/tree/master" target="_blank">common english words dataset</a></li>
</ul>
