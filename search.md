---
layout: page
permalink: /search/
---
  
#### Looking for something?  
  
{% include search.html %}

#### Search by category: 
{% for category in site.categories %}
  {% assign name = category[0] %}
  {% assign posts = category[1] %}
  <h4 id="{{ name | slugify }}">{{ name }}</h4>
  <ul>
    {% for post in posts %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
