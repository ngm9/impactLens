"""A data class representing the articles on internet"""
from attr import attrs, attrib


@attrs
class ArticleModel:
    text = attrib(default="")  # The document text string.
    url = attrib(default="")  # The URL for this article.
    headline = attrib(default="")  # The headline for the article.
