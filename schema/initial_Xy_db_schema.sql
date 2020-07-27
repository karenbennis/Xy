CREATE TABLE review_two(
	review_id VARCHAR NOT NULL,
	review_text TEXT NOT NULL,
	stars INT NOT NULL,
	cool INT NOT NULL,
	useful INT NOT NULL,
	funny INT NOT NULL,
	review_date date NOT NULL, 
	PRIMARY KEY (review_id)
);

CREATE TABLE business_two(
	review_id VARCHAR NOT NULL,
	business_id VARCHAR NOT NULL,
	FOREIGN KEY (review_id) REFERENCES review (review_id),
	PRIMARY KEY (review_id)
);

CREATE TABLE yelp_user_two(
	review_id VARCHAR NOT NULL,
	user_id VARCHAR NOT NULL,
	FOREIGN KEY (review_id) REFERENCES review (review_id),
	PRIMARY KEY (review_id)
);

CREATE TABLE review_class(
	review_id VARCHAR NOT NULL,
	class FLOAT,
	FOREIGN KEY (review_id) REFERENCES review (review_id),
	PRIMARY KEY (review_id)
);

SELECT 
	review_two.review_id,
	review_two.review_text,
	review_two.stars,
	review_two.cool,
	review_two.useful,
	review_two.funny,
	review_two.review_date,
	yelp_user_two.user_id
INTO user_review
FROM review_two
INNER JOIN yelp_user_two
ON review_two.review_id = yelp_user_two.review_id;
	
	
	
	