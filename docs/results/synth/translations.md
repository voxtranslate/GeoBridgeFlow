# GeoBridge-Flow: sample conversions (all through the hub; no image at test time)


## aud:sb -> text:sa

| source | reference | output |
|---|---|---|
| nótò kótà | sipa pise | sipa pise |
| nótò kótà mi líné víkè | sipa pise ge kufi kope | sipa pise ge kufi kope |
| líné víkè | kufi kope | kufi kope |
| nótò kótà | sipa pise | sipa pise |
| nábé bèdé mi nótò kótà | viba febu ge sipa pise | sipa febu ge tasa kope |
| nótò kótà mi nábé bèdé | sipa pise ge viba febu | sipa pise ge tebi nubu |
| nótò kótà | sipa pise | sipa pise |
| nótò kótà mi nábé bèdé | sipa pise ge viba febu | sipa pise ge tebi nubu |
| nótò kàvé mi nábé bèdé | vofu pise ge viba febu | tebi mipo ge viba kope |
| nótò kàvé | vofu pise | vofu pise ge vofu pise |
| nótò kàvé | vofu pise | vofu pise ge sipa pise |
| nótò kàvé | vofu pise | vofu pise |

## aud:sa -> text:sa

| source | reference | output |
|---|---|---|
| sipa pise | sipa pise | sipa pise |
| sipa pise ge kufi kope | sipa pise ge kufi kope | sipa pise ge vofu kope |
| kufi kope | kufi kope | kufi kope |
| sipa pise | sipa pise | sipa pise |
| viba febu ge sipa pise | viba febu ge sipa pise | viba febu ge sipa pise |
| sipa pise ge viba febu | sipa pise ge viba febu | sipa pise ge viba febu |
| sipa pise | sipa pise | sipa pise |
| sipa pise ge viba febu | sipa pise ge viba febu | sipa pise |
| vofu pise ge viba febu | vofu pise ge viba febu | vofu pise ge vofu febu |
| vofu pise | vofu pise | vofu pise |
| vofu pise | vofu pise | vofu pise |
| vofu pise | vofu pise | vofu pise |

## img:- -> text:sa

| source | reference | output |
|---|---|---|
| image im00003 | sipa pise | kufi pise |
| image im00003 | sipa pise ge kufi kope | kufi pise |
| image im00003 | kufi kope | kufi pise |
| image im00003 | sipa pise | kufi pise |
| image im00007 | viba febu ge sipa pise | sipa pise |
| image im00007 | sipa pise ge viba febu | sipa pise |
| image im00007 | sipa pise | sipa pise |
| image im00007 | sipa pise ge viba febu | sipa pise |
| image im00029 | vofu pise ge viba febu | viba febu ge viba febu |
| image im00029 | vofu pise | viba febu ge viba febu |
| image im00029 | vofu pise | viba febu ge viba febu |
| image im00029 | vofu pise | viba febu ge viba febu |

## txt:sa -> text:sc

| source | reference | output |
|---|---|---|
| sipa pise | तिदे फिकु | तिदे फिकु |
| sipa pise ge kufi kope | तिदे फिकु बो दोवो गेपा | तिदे फिकु बो दोवो गुको |
| kufi kope | दोवो गेपा | दोवो गेपा |
| sipa pise | तिदे फिकु | तिदे फिकु |
| viba febu ge sipa pise | लादे मावि बो तिदे फिकु | लादे मावि बो तिदे फिकु |
| sipa pise ge viba febu | तिदे फिकु बो लादे मावि | तिदे फिकु बो लादे मावि |
| sipa pise | तिदे फिकु | तिदे फिकु |
| sipa pise ge viba febu | तिदे फिकु बो लादे मावि | तिदे फिकु बो लादे मावि |
| vofu pise ge viba febu | विला फिकु बो लादे मावि | विला फिकु बो लादे मावि |
| vofu pise | विला फिकु | विला फिकु |
| vofu pise | विला फिकु | विला फिकु |
| vofu pise | विला फिकु | विला फिकु |

## txt:sc -> text:sa

| source | reference | output |
|---|---|---|
| तिदे फिकु | sipa pise | sipa pise |
| तिदे फिकु बो दोवो गेपा | sipa pise ge kufi kope | sipa pise ge kufi kope |
| दोवो गेपा | kufi kope | kufi kope |
| तिदे फिकु | sipa pise | sipa pise |
| लादे मावि बो तिदे फिकु | viba febu ge sipa pise | viba febu ge sipa pise |
| तिदे फिकु बो लादे मावि | sipa pise ge viba febu | sipa pise ge viba febu |
| तिदे फिकु | sipa pise | sipa pise |
| तिदे फिकु बो लादे मावि | sipa pise ge viba febu | sipa pise ge viba febu |
| विला फिकु बो लादे मावि | vofu pise ge viba febu | vofu pise ge vofu febu |
| विला फिकु | vofu pise | vofu pise |
| विला फिकु | vofu pise | vofu pise |
| विला फिकु | vofu pise | vofu pise |
