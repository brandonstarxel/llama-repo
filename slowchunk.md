Issues with slow chunking:

I am sure it will fail to find the chunks. Why? GPT won't exactly copy the chunks since it'll correct tiny bits. 


Hmm okay I do want to have SOMETHING to show for today. Let's do fast chunking then since that will give results. 

GPT Chunking:

Request it respond in JSON and give reason for why it chunks. 

Happy to continue with current chunk tagging and re-connecting.

Next steps:

1). Pull metric code from repo and test it works with basic chunker. 
2). Turn fast chunker into Chroma style chunker so we can score it. 
3). Run again with GPT.

Try JSON based:

{
    "chunk_id": 4,
    "reason": "The chunks change topic to ... after this one." 
}