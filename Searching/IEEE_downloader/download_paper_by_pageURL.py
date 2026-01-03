# -*- coding: utf-8 -*-
# @Time    : 2021/10/13 10:37 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import json
import requests
import os
from utils import downLoad_paper
import re



def organize_info_by_query(queryText, pageNumber, save_dir, paper_name_with_year=None):
    # Use a requests Session to get homepage cookies (more robust than urllib)
    session = requests.Session()
    try:
        r_home = session.get("https://ieeexplore.ieee.org", timeout=10,
                             headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        cookies = session.cookies.get_dict()
        cookie_valid = "; ".join([f"{k}={v}" for k, v in cookies.items()]) if cookies else ""
    except Exception as e:
        print(f"[Warning] failed to fetch IEEE homepage cookies: {e}")
        cookie_valid = ""
    paper_info = {}
    count = 0
    for page in pageNumber:
        headers = {
            'Host': 'ieeexplore.ieee.org',
            'Content-Type': "application/json",
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'Cookie': cookie_valid,
            'Referer': 'https://ieeexplore.ieee.org',
            'Accept': 'application/json, text/plain, */*'
        }
        payload = {"queryText": queryText, "pageNumber": str(page), "returnFacets": ["ALL"],
                   "returnType": "SEARCH"}

        # Try POST with session and simple retry/backoff for transient blocks
        toc_res = None
        for attempt in range(3):
            try:
                toc_res = session.post("https://ieeexplore.ieee.org/rest/search", headers=headers,
                                       data=json.dumps(payload), timeout=15)
                if toc_res.status_code == 200:
                    break
                else:
                    # 418 or other codes may indicate blocking; wait and retry
                    print(f"[Warning] status_code={toc_res.status_code}; attempt={attempt+1}")
                    time.sleep(2 + attempt * 2)
            except Exception as e:
                print(f"[Warning] request attempt {attempt+1} failed: {e}")
                time.sleep(2 + attempt * 2)

        if toc_res is None:
            print("[Error] search request failed after retries")
            continue

        try:
            response = json.loads(toc_res.text)
        except Exception as e:
            print(f"[Error] failed to parse search response: {e}")
            continue
        if 'records' in response:
            for item in response['records']:
                paper_info[count] = {}
                paper_info[count]['url'] = "https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=" + item['articleNumber'] + "&ref="
                paper_info[count]['name'] = item['articleTitle']
                rstr = r"[\=\(\)\,\/\\\:\*\?\？\"\<\>\|\'']"
                if paper_name_with_year:
                    paper_info[count]['name'] = os.path.join(save_dir, item['publicationYear'] + ' ' + re.sub(rstr, '', paper_info[count]['name']) + '.pdf')
                else:
                    paper_info[count]['name'] = os.path.join(save_dir, re.sub(rstr, '', paper_info[count]['name']) + '.pdf')
                count += 1
    if len(paper_info) > 0:
        return True, paper_info
    else:
        return False, paper_info


if __name__ == '__main__':
    import utils
    utils._init()
    queryText = "dialog system"
    pageNumber = [3]
    save_dir = "save"
    _, paper_info = organize_info_by_query(queryText, pageNumber, save_dir, True)
    downLoad_paper(paper_info)
