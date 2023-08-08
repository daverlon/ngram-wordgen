<h1>Word generator using N-gram probabilities</h1>

<p>Word generator using <a href="https://en.wikipedia.org/wiki/N-gram">N-gram</a> probabilities written in Python using only numpy.</p>

<p>Based on <a href="https://github.com/karpathy/makemore" target="_blank">Karpathy's</a> bigram model shown in his makemore lectures.</p>

<p>Example usage:</p>
<pre><p>python3 ngram.py -fdatasets/names.txt -n6 -N30 --skip-existing --show-existing |> out.txt</p></pre>
<ul>
    <li>-fdatasets/names.txt (use the file located at ./datasets/names.txt) - <b>required</b></li>
    <li>-n6 (6-gram model) - <b>required</b></li>
    <li>-N30 (generate 30 words) - <b>required</b></li>
    <li>--skip-existing (skip word generations that already exist in the dataset) - optional</li>
    <li>--show-existing (display "✓" after words that already exist in the dataset) - optional</li>
    <li>|> out.txt (save generated words into a file, e.g. out.txt) - optional</li>
</ul>

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
    <s><li>Option to only count non-existing unique generations</li></s>
    <s><li>Automatic <b>n-gram</b> probability distributions generator</li></s>
</ul>

<br>

<p>Thanks:</p>
<ul>
<li><a href="https://www.ssa.gov/oact/babynames/" target="_blank">SSA.gov baby names dataset</a>
<li><a href="https://github.com/first20hours/google-10000-english/tree/master" target="_blank">common english words dataset</a></li>
</ul>
