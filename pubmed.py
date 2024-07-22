import re
import json
import time
import xmltodict
import urllib.error
import urllib.parse
import urllib.request
from tqdm import tqdm
from typing import Iterator, List, Optional
from langchain.schema import Document
from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool
from langchain.pydantic_v1 import BaseModel
from langchain.callbacks.manager import CallbackManagerForToolRun


class PubMedAPIWrapper(BaseModel):
    """
    Wrapper around PubMed API.

    This wrapper will use the PubMed API to conduct searches and fetch
    document summaries. By default, it will return the document summaries
    of the top-k results of an input search.

    Parameters:
        MAX_QUERY_LENGTH: maximum length of the query.
          Default is 300 tokens.
        max_retry: maximum number of retries for a request. Default is 5.
        sleep_time: time to wait between retries.
          Default is 0.2 seconds.
    """

    base_url_esearch: str = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    )
    base_url_efetch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    base_url_citation: str = "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/?"
    max_retry: int = 5
    sleep_time: float = 30

    # Default values for the parameters
    MAX_QUERY_LENGTH: int = 128

    def run(self, query: str, top_k_results: int = 10) -> List:
        """
        Run PubMed search and get the article meta information.
        See https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
        It uses only the most informative fields of article meta information.
        """
        query = re.sub('\s+', '+', query)
        # Retrieve the top-k results for the query
        docs = list()
        for result in self.load(query[:self.MAX_QUERY_LENGTH], top_k_results):
            if result['pmc_id'] is None:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{result['uid']}/"
            else:
                url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{result['pmc_id']}/"
            docs.append({'URL': url, 'Summary': result['Summary'], 'Citation': result['Citation']})
        return docs

    def lazy_load(self, query: str, top_k_results: int) -> Iterator[dict]:
        """
        Search PubMed for documents matching the query.
        Return an iterator of dictionaries containing the document metadata.
        """
        url = (
                self.base_url_esearch
                + "db=pubmed&term="
                + str(urllib.parse.quote(query, safe=':/?=&"[]+'))
                + f"&retmode=json&retmax=1000&usehistory=y"
                + "&sort=relevance"
        )
        result = urllib.request.urlopen(url)
        text = result.read().decode("utf-8")
        json_text = json.loads(text)
        webenv = json_text["esearchresult"]["webenv"]
        references = list()
        for uid in tqdm(json_text["esearchresult"]["idlist"][:top_k_results]):
            try:
                details = self.retrieve_article(uid, webenv)
                if details['pmc_id'] is None:
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{details['uid']}/"
                else:
                    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{details['pmc_id']}/"
                references.append(
                    {'title': details['title'],
                     'source_type': 'pubmed',
                     'abstract': details['Summary'],
                     'url': url,
                     'current_page': 1})
                yield details
            except:
                continue

    def load(self, query: str, top_k_results: int) -> List[dict]:
        """
        Search PubMed for documents matching the query.
        Return a list of dictionaries containing the document metadata.
        """
        return [x for x in self.lazy_load(query, top_k_results) if len(x['Summary'].strip()) > 0]

    def _dict2document(self, doc: dict) -> Document:
        summary = doc.pop("Summary")
        return Document(page_content=summary, metadata=doc)

    def lazy_load_docs(self, query: str) -> Iterator[Document]:
        for d in self.lazy_load(query=query):
            yield self._dict2document(d)

    def load_docs(self, query: str) -> List[Document]:
        return list(self.lazy_load_docs(query=query))

    def retrieve_article(self, uid: str, webenv: str) -> dict:
        url = (
                self.base_url_efetch
                + "db=pubmed&retmode=xml&id="
                + uid
                + "&webenv="
                + webenv
        )
        retry = 0
        while True:
            try:
                result = urllib.request.urlopen(url)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.max_retry:
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    print(
                        f"Too Many Requests, "
                        f"waiting for {self.sleep_time:.2f} seconds..."
                    )
                    time.sleep(self.sleep_time)
                    self.sleep_time *= 2
                    retry += 1
                else:
                    raise e
        xml_text = result.read().decode("utf-8")
        text_dict = xmltodict.parse(xml_text)
        return self._parse_article(uid, text_dict)

    def _parse_article(self, uid: str, text_dict: dict) -> dict:
        try:
            ar = text_dict["PubmedArticleSet"]["PubmedArticle"]["MedlineCitation"][
                "Article"
            ]
        except KeyError:
            ar = text_dict["PubmedArticleSet"]["PubmedBookArticle"]["BookDocument"]
        abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_text, str):
            summary = abstract_text
        elif isinstance(abstract_text, dict):
            summary = abstract_text.get('#text', '')
        else:
            summaries = []
            for txt in abstract_text:
                prefix = ''
                if '@NlmCategory' in txt:
                    prefix += (txt['@NlmCategory'] + ': ')
                if '@Label' in txt:
                    prefix += (txt['@Label'] + ': ')
                if '#text' in txt:
                    summaries.append(prefix + txt['#text'])
            summary = '\n'.join(summaries)
        try:
            article_ids = text_dict["PubmedArticleSet"]["PubmedArticle"]["PubmedData"]["ArticleIdList"]['ArticleId']
            pmc_ids = [x['#text'] for x in article_ids if x['@IdType'] == 'pmc']
            if len(pmc_ids) > 0:
                pmc_id = pmc_ids[0]
            else:
                pmc_id = None
        except:
            pmc_id = None
        citation_url = self.base_url_citation + "format=citation&id=" + uid
        result = urllib.request.urlopen(citation_url)
        citations = json.loads(result.read().decode("utf-8"))
        if isinstance(ar['ArticleTitle'], str):
            title = ar['ArticleTitle']
        else:
            title = ar['ArticleTitle']['#text']
        return {
            "uid": uid,
            "pmc_id": pmc_id,
            "Summary": summary,
            "Citation": citations['ama']['orig'],
            'title': title
        }


class PubmedQueryRun(BaseTool):
    """Tool that searches the PubMed API."""

    name: str = "PubMed"
    description: str = (
        "A wrapper around PubMed database. "
        "Useful for when you need to answer questions about medicine, health, and biomedical topics "
        "from biomedical literature, MEDLINE, life science journals, and online books. "
        "Input should be a search query in English. "
        "Please use PubMed advanced search expression and syntax to build complex query."
        "Don't use quotation marks in query term. Field name is needed after date condition"
    )
    api_wrapper: PubMedAPIWrapper = Field(default_factory=PubMedAPIWrapper)
    top_k_results: int = 5
    doc_content_tokens_max: int = 2000

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the PubMed tool."""
        docs = self.api_wrapper.run(query, self.top_k_results)
        docs = [
            f"URL: {result['URL']}\n"
            f"Summary:\n{result['Summary'][:self.doc_content_tokens_max]}\n"
            f"Citation:\n{result['Citation']}"
            for result in docs
        ]
        return "\n\n".join(docs) if docs else "No good PubMed Result was found"


class PubmedQueryRunList(BaseTool):
    """Tool that searches the PubMed API."""

    name: str = "PubMed"
    description: str = (
        "A wrapper around PubMed database. "
        "Useful for when you need to answer questions about medicine, health, and biomedical topics "
        "from biomedical literature, MEDLINE, life science journals, and online books. "
        "Input should be a search query in English. "
        "Please use PubMed advanced search expression and syntax to build complex query."
        "Don't use quotation marks in query term. Field name is needed after date condition"
    )
    api_wrapper: PubMedAPIWrapper = Field(default_factory=PubMedAPIWrapper)

    def _run(
            self,
            query: str,
            top_k_results: int = 5,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List:
        """Use the PubMed tool."""
        docs = self.api_wrapper.run(query, top_k_results)
        return docs
