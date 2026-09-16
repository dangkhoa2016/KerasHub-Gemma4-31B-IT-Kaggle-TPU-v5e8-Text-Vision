# Clients

Python:

```bash
python clients/python/gemma4_client.py \
  --api-key "$API_KEY" \
  --prompt "Explain TPU model parallelism."
```

Node.js 18+:

```js
import { generate } from "./clients/node/gemma4-client.mjs";
console.log(await generate("Explain TPU model parallelism."));
```
