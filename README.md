<h1>Name generation with N-gram probabilities</h1>

<p>Name generator using <a href="https://en.wikipedia.org/wiki/N-gram">N-gram</a> probabilities written in Python using only numpy.</p>

<p>Based on <a href="https://github.com/karpathy/makemore" target="_blank">Karpathy's</a> bigram model shown in his makemore lectures.</p>

<p>Example usage:</p>
<pre><p>python3 ngram.py -n6 -N30</p></pre>
<ul>
    <li>-n6 (6-gram model)</li>
    <li>-N30 (generate 30 names)</li>
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
    <s><li>Automatic <b>n-gram</b> probability distributions generator</li></s>
</ul>