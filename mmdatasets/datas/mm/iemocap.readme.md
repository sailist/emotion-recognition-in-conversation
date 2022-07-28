## Label Distribution

After considering multiple labels like `Ses01F_impro01_F005 :Frustration; :Anger; ()`

I got the label distribution results below:

```
Counter({'Neutral': 1726,
         'Frustration': 2916,
         'Anger': 1269,
         'Sadness': 1251,
         'Happiness': 656,
         'Excited': 1976,
         'Surprise': 110,
         'Fear': 107,
         'Other': 26,
         'Disgust': 2})
```

## Error

For example, file Ses01F_impro02.txt contains text like:

```
Ses01F_impro02_F000 [007.2688-016.6000]: Did you get the mail? So you saw my letter?
M: Yeah.
Ses01F_impro02_M000 [017.6000-020.6264]: It's not fair.
```

identity and time segment are not labeled, which result in difference number between labels and sentends.

Finally, I got 11268 labels, but only 10085 sentents,and there

Secondly, there are some transcripts like

```
> Session3/dialog/transcriptions/Ses03F_impro05.txt
Ses03F_impro05_MXX0 [151.2551-153.4251]: [LAUGHTER], That's what they say.
```

These sentents have no labels. All these sentents have identity like MXX[number] or FXX[number]. (while others has
M[number])


