#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, json, time
import urllib.request, urllib.error
from pathlib import Path

class Gemma4Client:
    def __init__(
        self,
        base_url,
        api_key,
        request_timeout=30,
        poll_timeout=1800,
    ):
        self.base_url=base_url.rstrip("/")
        self.api_key=api_key
        self.request_timeout=request_timeout
        self.poll_timeout=poll_timeout

    def _request(
        self,
        method,
        path,
        payload=None,
        timeout=None,
    ):
        data=None
        headers={
            "Authorization":
            f"Bearer {self.api_key}"
        }
        if payload is not None:
            data=json.dumps(payload).encode()
            headers["Content-Type"]="application/json"

        req=urllib.request.Request(
            self.base_url+path,
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(
                req,
                timeout=(
                    timeout
                    or self.request_timeout
                ),
            ) as response:
                return (
                    response.status,
                    json.loads(
                        response.read().decode()
                    ),
                )
        except urllib.error.HTTPError as exc:
            body=exc.read().decode()
            try:
                parsed=json.loads(body)
            except Exception:
                parsed={"error":body}
            return exc.code,parsed

    def _poll(self, job_id):
        deadline=(
            time.monotonic()
            + self.poll_timeout
        )
        while time.monotonic()<deadline:
            status,body=self._request(
                "GET",
                f"/result/{job_id}",
            )
            if status==200:
                return body
            if (
                status>=400
                and status!=202
            ):
                raise RuntimeError(body)
            time.sleep(1)
        raise TimeoutError(
            f"Timed out waiting for {job_id}"
        )

    def generate(
        self,
        prompt,
        system="",
        max_new_tokens=128,
    ):
        status,body=self._request(
            "POST",
            "/generate/async",
            {
                "prompt":prompt,
                "system":system,
                "max_new_tokens":
                    max_new_tokens,
            },
        )
        if status!=202:
            raise RuntimeError(body)
        return self._poll(body["job_id"])

    def generate_image(
        self,
        image_path,
        prompt,
        system="",
        max_new_tokens=128,
    ):
        encoded=base64.b64encode(
            Path(image_path).read_bytes()
        ).decode()
        status,body=self._request(
            "POST",
            "/generate/image/async",
            {
                "prompt":prompt,
                "system":system,
                "max_new_tokens":
                    max_new_tokens,
                "image_base64":encoded,
            },
        )
        if status!=202:
            raise RuntimeError(body)
        return self._poll(body["job_id"])

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument(
        "--url",
        default="http://127.0.0.1:7860",
    )
    p.add_argument(
        "--api-key",
        required=True,
    )
    p.add_argument(
        "--prompt",
        required=True,
    )
    p.add_argument(
        "--system",
        default="",
    )
    p.add_argument("--image")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    a=p.parse_args()
    client=Gemma4Client(
        a.url, a.api_key
    )
    result=(
        client.generate_image(
            a.image,
            a.prompt,
            a.system,
            a.max_new_tokens,
        )
        if a.image
        else client.generate(
            a.prompt,
            a.system,
            a.max_new_tokens,
        )
    )
    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))
