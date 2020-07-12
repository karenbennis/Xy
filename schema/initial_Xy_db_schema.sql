CREATE TABLE review(
	review_id VARCHAR NOT NULL,
	review_text TEXT NOT NULL,
	stars INT NOT NULL,
	cool INT NOT NULL,
	useful INT NOT NULL,
	funny INT NOT NULL,
	review_date date NOT NULL, 
	review_type varchar NOT NULL,
	PRIMARY KEY (review_id)
);

CREATE TABLE business(
	review_id VARCHAR NOT NULL,
	business_id VARCHAR NOT NULL,
	FOREIGN KEY (review_id) REFERENCES review (review_id),
	PRIMARY KEY (review_id)
);

CREATE TABLE yelp_user(
	review_id VARCHAR NOT NULL,
	user_id VARCHAR NOT NULL,
	FOREIGN KEY (review_id) REFERENCES review (review_id),
	PRIMARY KEY (review_id)
);